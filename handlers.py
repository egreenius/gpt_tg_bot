from telegram import Update
from telegram.ext import CallbackContext

from config import Start_Text
from utils import SunTgBot


classBot = SunTgBot()


# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
# Обработка сообщения, содержащего команду /start.

def start(update: Update, context: CallbackContext):
    # update.message.reply_text('Hi!')
    # context.bot.send_message(chat_id=update.effective_chat.id,
    #                          text="Привет!")
    context.bot.send_message(chat_id=update.message.chat_id,
                             text=Start_Text)


# Обработка сообщения от пользователя
def text_message(update: Update, context: CallbackContext):
    txt = update.message.text
    user = update.message.from_user.id
    user_name = update.message.from_user.username
    print(f'User: {user}, Name: {user_name}')
    context.bot.send_chat_action(chat_id=update.message.chat_id, action='typing')
    intention = classBot.choose_intention(txt)
    print(f'Intention: {intention}')
    if intention == 'simpleTalk':
        result = classBot.generate_text_simpletalk(txt, user)
    elif intention == 'Кулинарные рецепты':
        result = classBot.generate_text_recipe(txt, intention)
        # result = 'Кулинарные рецепты'
    elif intention == 'Погода':
        result = classBot.generate_text_weather(txt)
        # result = 'Прогноз погоды'
    elif intention == 'грипп':
        result = classBot.generate_text_health(txt, intention)
        # result = 'лечение'
    elif intention == 'Прочее':
        result = classBot.generate_text_talk(txt, user, intention)
        # result = 'Прочее'
    else:
        result = 'Я тебя тоже люблю!'
    # result = respond([txt])
    if result:
        context.bot.send_message(chat_id=update.message.chat_id,
                                 text=result)
    else:
        context.bot.send_message(chat_id=update.message.chat_id,
                                 text='Привет! Привет!')
