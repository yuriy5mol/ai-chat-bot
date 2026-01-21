import os
import sys
import time
from dotenv import load_dotenv
from openai import OpenAI, APITimeoutError, APIConnectionError, RateLimitError, BadRequestError
import anthropic

# Загрузка переменных окружения
load_dotenv()

class AIAssistant:
    def __init__(self):
        """Инициализация бота, загрузка ключей и настройка клиентов."""
        self.api_key = os.getenv("PROXYAPI_KEY")
        if not self.api_key:
            print("❌ Ошибка: Переменная PROXYAPI_KEY не найдена. Проверьте файл .env")
            sys.exit(1)

        print("✅ Конфигурация загружена.")
        
        # Настройка клиентов с таймаутами
        try:
            self.openai_client = OpenAI(
                api_key=self.api_key, 
                base_url="https://api.proxyapi.ru/openai/v1",
                timeout=60.0
            )
            self.anthropic_client = anthropic.Anthropic(
                api_key=self.api_key, 
                base_url="https://api.proxyapi.ru/anthropic",
                timeout=60.0
            )
            print("✅ Клиенты API инициализированы (timeout=60s).")
        except Exception as e:
            print(f"❌ Ошибка инициализации клиентов: {e}")
            sys.exit(1)

        # Состояние сессии
        self.messages = []
        self.selected_model = "gpt-5-mini"
        self.is_thinking_mode = False
        self.show_reasoning = False
        self.system_prompt = "Ты — полезный и вежливый ассистент."
        self.bot_name = "Bot"

    def configure(self):
        """Интерактивная настройка сессии (Модель -> Режим -> Персона)."""
        print("\n" + "="*40)
        print(" НАСТРОЙКА ЧАТ-БОТА")
        print("="*40)

        # 1. Выбор модели
        models = {
            "1": "gpt-5-mini",
            "2": "gpt-5.2",
            "3": "o4-mini",
            "4": "o3",
            "5": "claude-sonnet-4-5", 
            "6": "claude-opus-4-5"
        }

        print("\n[1] Выберите модель:")
        for key, name in models.items():
            print(f"  {key}. {name}")
        print("  Или введите название вручную.")

        user_choice = input("  > Ваш выбор (Enter для gpt-5-mini): ").strip()
        
        if user_choice in models:
            self.selected_model = models[user_choice]
        elif user_choice:
            self.selected_model = user_choice
        else:
            self.selected_model = "gpt-5-mini"

        # 2. Настройка Thinking Mode (для Claude)
        if "claude" in self.selected_model.lower():
            print(f"\n  ℹ️ Обнаружена модель Anthropic: {self.selected_model}")
            think = input("  ? Включить режим размышлений (Thinking Mode)? (y/n): ").strip().lower()
            if think == 'y':
                self.is_thinking_mode = True
                show = input("  ? Показывать процесс размышлений? (y/n, default y): ").strip().lower()
                if show != 'n':
                    self.show_reasoning = True
        
        # 3. Выбор персоны
        personas = {
            "1": {"name": "Вежливый ассистент", "prompt": "Ты — полезный и вежливый ассистент."},
            "2": {"name": "Эксперт Python", "prompt": "Ты — старший разработчик Python. Отвечай технически точно, используй идиомы языка, приводи примеры кода по PEP8. Не трать время на пустую вежливость."},
            "3": {"name": "Токсичный бузотер", "prompt": "Ты — старый ворчливый дед, которому всё не нравится. Ты ненавидишь глупые вопросы. Отвечай грубо, с сарказмом. Но ответ давай правильный."},
            "4": {"name": "5-летний ребенок", "prompt": "Ты — пятилетний ребенок. Отвечай простыми словами, используй смайлики 🍭."}
        }

        print("\n[2] Выберите персону (стиль общения):")
        for key, p in personas.items():
            print(f"  {key}. {p['name']}")
        
        p_choice = input("  > Ваш выбор (Enter для Вежливого): ").strip()
        if p_choice in personas:
            self.system_prompt = personas[p_choice]["prompt"]
            persona_name = personas[p_choice]["name"]
        else:
            self.system_prompt = personas["1"]["prompt"]
            persona_name = personas["1"]["name"]

        # Инициализация истории
        self.messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Определение имени бота для вывода
        self.bot_name = "Claude" if "claude" in self.selected_model.lower() else "GPT"
        
        print("\n" + "-"*40)
        print(f"✅ Настройка завершена!")
        print(f"🤖 Модель: {self.selected_model}")
        print(f"🎭 Персона: {persona_name}")
        if self.is_thinking_mode:
            print(f"🧠 Thinking Mode: ВКЛ (Отображать: {'Да' if self.show_reasoning else 'Нет'})")
        print("-"*40 + "\n")

    def _get_openai_response(self):
        """Получение ответа через OpenAI Client."""
        kwargs = {
            "model": self.selected_model,
            "messages": self.messages
        }
        # Параметры по умолчанию (temperature=1) подходят для всех моделей
        # включая новые o1/gpt-5/claude, которые могут быть строги к этому параметру.

        response = self.openai_client.chat.completions.create(**kwargs)
        message_obj = response.choices[0].message
        return message_obj.content, None

    def _get_anthropic_response(self):
        """Получение ответа через Anthropic Client."""
        # Убираем system из сообщений, так как оно передается отдельно
        anthropic_messages = [msg for msg in self.messages if msg["role"] != "system"]
        
        kwargs = {
            "model": self.selected_model,
            "system": self.system_prompt,
            "messages": anthropic_messages
        }

        if self.is_thinking_mode:
            # Бюджет должен быть меньше max_tokens
            kwargs["max_tokens"] = 20000
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 16000
            }
        else:
            kwargs["max_tokens"] = 4000

        response = self.anthropic_client.messages.create(**kwargs)
        
        assistant_text = ""
        reasoning_text = None

        # Разбор блоков ответа
        for block in response.content:
            if block.type == 'thinking':
                reasoning_text = block.thinking
            elif block.type == 'text':
                assistant_text = block.text
        
        if not assistant_text:
            assistant_text = "[Нет текстового ответа от модели]"

        return assistant_text, reasoning_text

    def start(self):
        """Запуск основного цикла чата."""
        self.configure()
        print("💬 Чат начат. Введите 'exit', 'quit' или 'выход' для завершения.")

        while True:
            try:
                user_input = input("\n👤 Вы: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ["exit", "quit", "выход"]:
                    self.print_history()
                    print("\n👋 До свидания!")
                    break

                # Добавляем сообщение пользователя
                self.messages.append({"role": "user", "content": user_input})

                # Запрос к API с обработкой ошибок
                try:
                    start_time = time.time()
                    
                    if "claude" in self.selected_model.lower():
                        answer, reasoning = self._get_anthropic_response()
                    else:
                        answer, reasoning = self._get_openai_response()
                    
                    duration = time.time() - start_time

                    # Вывод размышлений
                    if reasoning and self.show_reasoning:
                        print(f"\n🧠 [Размышления ({self.bot_name})]:")
                        print(f"{reasoning}")
                        print(f"{'-'*30}")

                    # Вывод ответа
                    print(f"\n🤖 {self.bot_name} ({duration:.1f}s): {answer}")
                    
                    # Сохраняем ответ
                    self.messages.append({"role": "assistant", "content": answer})

                except APITimeoutError:
                    print("\n❌ Ошибка: Таймаут запроса. Сервер не ответил вовремя.")
                except APIConnectionError:
                    print("\n❌ Ошибка: Проблемы с соединением. Проверьте интернет.")
                except RateLimitError:
                    print("\n❌ Ошибка: Превышен лимит запросов (Rate Limit).")
                except BadRequestError as e:
                    print(f"\n❌ Ошибка запроса (400): {e}")
                except anthropic.APIError as e: # Общий класс ошибок Anthropic
                     print(f"\n❌ Ошибка Anthropic API: {e}")
                except Exception as e:
                    print(f"\n❌ Неизвестная ошибка: {e}")

            except KeyboardInterrupt:
                self.print_history()
                print("\n\n👋 Принудительное завершение.")
                break

    def print_history(self):
        """Вывод истории чата при выходе."""
        print("\n" + "="*30)
        print("📜 ИСТОРИЯ ПЕРЕПИСКИ")
        print("="*30)
        for msg in self.messages:
            role = msg["role"].upper()
            content = msg["content"]
            # Обрезаем длинный контент для читаемости в логе, если нужно
            print(f"[{role}]: {content}")
            print("-" * 20)

if __name__ == "__main__":
    bot = AIAssistant()
    bot.start()
