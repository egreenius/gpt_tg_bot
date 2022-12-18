import logging
import os
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from handlers import start, text_message
from keys import TOKEN


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # определение порта
    PORT = int(os.environ.get('PORT', '8443'))

    # создание экземпляра объекта Updater для получения обновлений от телеграма и передачи их в Dispatcher
    updater = Updater(TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    # обработка полученных из Updater сообщений
    # Обработка команды /start c помощью CommandHandler (он среагирует на команду /start)
    # и передача в нашу функцию обработки
    dispatcher.add_handler(CommandHandler("start", start))
    # Обработка обычного сообщения, пропущенного через фильтр (только текст и команды)
    # и передача в нашу функцию обработки
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, text_message))

    # далее или поллинг
    updater.start_polling()

    # или вебхоок. Что то одно должно быть
    # add handlers
    # updater.start_webhook(listen="0.0.0.0",
    #                       port=PORT,
    #                       url_path=TOKEN,
    #                       webhook_url="https://hidden-mesa-48772.herokuapp.com/" + TOKEN)

    updater.idle()


if __name__ == '__main__':
    main()
