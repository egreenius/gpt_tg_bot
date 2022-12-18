import random
import re
import sys

import math
import pymorphy2
import requests
import torch
from pullenti_wrapper.processor import Processor, GEO
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from config import intentionList, weatherParams, url_weather, simpleTalk, model


# функция очистки входного текста
def clear_text(input_text):
    # заменяем во входном тексте все, кроме букв или цифр на пробел
    input_text = re.sub(r"\W", " ", input_text)
    # переводим в нижний регистр и обрезаем лишние пробелы
    txt = str(input_text).lower().strip()
    return txt


def format_json(inp_json, reqType, place):
    if reqType == 'weather':
        result = f'\n Погода в {place} {inp_json["name"]}, {inp_json["weather"][0]["description"]}, температура ' \
                 f' {round(inp_json["main"]["temp_min"])}° ... {round(inp_json["main"]["temp_max"])}°, ощущается как ' \
                 f'{round(inp_json["main"]["feels_like"])}°'
    else:
        result = ''
    return result


# def get_geo(referent):
#     if referent.type != 'государство':
#         name_list = []
#         for slot in referent.slots:
#             if slot.key == 'NAME':
#                 name_list.append(slot.value)
#         result = min(name_list, key=len)
#     else:
#         result = referent.name
#     return result, referent.type


class SunTgBot:
    """
    Class for generating text with pretrained ruGPT3

    Vars:
    ----
    inputText - текст на входе, для которого нужно определеить намерение или продолжить

    Attributes:
    ----------
    __intentionList: List - список "намерений", точнее подготовленные подсказки для намерений
    __tokenizer: GPT2Tokenizer - используемый токенайзер (обычно той же модели)
    __model: GPT2LMHeadModel - используемая gpt модель

    Methods:
    --------
    choose_intention(inputText) - определение намерений по входной фразе
    generate_text_recipe(inputText, intention) - генерация текста по входной фразе и намерению
    generate_text_weather(inputText, intention)  - генерация текста о погоде в каком то населенном пункте
    generate_text_talk(inputText, intention) - генерация текста для болталки на общие темы
    generate_text_simpletalk(inputText) - генерация текста для быстрых ответов на частые вопросы
    generate_text_health(self, inputText, intention) - генерция текста по лечению

    """

    def __init__(self) -> None:
        if len(sys.argv) > 1:
            model_name = model.get(sys.argv[1], 'sberbank-ai/rugpt3medium_based_on_gpt2')
        else:
            model_name = 'sberbank-ai/rugpt3medium_based_on_gpt2'
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.__model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained(model_name).to(self.__device)
        self.__NER: Processor = Processor([GEO])
        self.__intentionList = intentionList
        self.__weatherParams = weatherParams
        self.__weatherUrl = url_weather
        self.__morph = pymorphy2.MorphAnalyzer()
        self.history = dict()
        print(f'Model: {model_name}')

    @staticmethod
    def get_prompt(intention: str) -> str:
        if intention == 'Кулинарные рецепты':
            prompt = 'Готовим:'  # Приготовим
        elif intention == 'грипп':
            prompt = ''  # чтобы быть здоровым нужно:
        else:
            prompt = ''
            # elif intention == 'Прочее':
            #     prompt = 'дядь, дай десять копеек! \nМожет тебе еще ключит дать, от квартиры где деньги лежат?\n' # ОБ

        return prompt

    @staticmethod
    def get_postfix(intention: str) -> tuple:
        if intention == 'Кулинарные рецепты':
            end_token = 'Приятного аппетита!'
            postfix = '\nПриятного аппетита!'
        elif intention == 'грипп':
            end_token = '<s>'
            postfix = '\nНе болей!'
        elif intention == 'Прочее':
            end_token = '<s>'
            postfix = ''
        else:
            end_token = ''
            postfix = ''
        return end_token, postfix

    def get_ppl(self, sentence) -> float:
        # функция получения перплексии для переданного текста
        # sentence - текст, для которого определяется перплексия
        encodings = self.__tokenizer(sentence, return_tensors='pt')
        input_ids = encodings.input_ids.to(self.__device)
        with torch.no_grad():
            outputs = self.__model(
                input_ids=input_ids,
                labels=input_ids
            )
        loss = outputs.loss
        return math.exp(loss.item() * input_ids.size(1))

    def choose_intention(self, inputText) -> str:
        # будущий список измеренных перплексий входного текста для каждого намерения
        s_txt = clear_text(inputText)
        if s_txt in simpleTalk:
            result = 'simpleTalk'
        else:
            ppl_values = []
            for intention in self.__intentionList:
                sentence = intention + ':' + inputText
                ppl = self.get_ppl(sentence)
                ppl_values.append(ppl)
            # пронумеруем список, в итоге получится список кортежей
            ppl_num = list(enumerate(ppl_values, 0))
            min_ppl = min(ppl_num, key=lambda i: i[1])
            result = self.__intentionList[min_ppl[0]]
        return result

    def generate_text_recipe(self, inputText, intention) -> str:
        text = SunTgBot.get_prompt(intention) + inputText
        tokens = self.__tokenizer(text, return_tensors='pt').to(self.__device)
        size = tokens['input_ids'].shape[1]

        output = self.__model.generate(
            **tokens,
            do_sample=False,
            max_length=size + 350,
            early_stopping=True,
            repetition_penalty=8.,  # 8.
            temperature=1,  # 0.8
            num_beams=10,
        )

        decoded = self.__tokenizer.decode(output[0])
        result = decoded[len(text):]
        print(f'Recipe: {result}')
        if result:
            if result.find(SunTgBot.get_postfix(intention)[0]) >= 0:
                result = result[:result.find(SunTgBot.get_postfix(intention)[0])]
                if result.find('<s>') >= 0:
                    # возвращаем одну из строк, разделенную <s>
                    result = random.sample(result.split(sep='<s>')[:-1], k=1)[0]
                    if result.find(text.lower()) >= 0:
                        result = result.split(text.lower())[1]
            elif result.find('<s>') >= 0:
                # возвращаем одну из строк, разделенную <s>
                result = random.sample(result.split(sep='<s>')[:-1], k=1)[0]
                if result.find(text.lower()):
                    result = result.split(text.lower())[1]
            else:
                result = result + '...'
        else:
            result = '\nхм... пожалуй здесь я тебе не помогу, разве что пожелаю '
        return inputText + result + SunTgBot.get_postfix(intention)[1]

    def generate_text_health(self, inputText, intention) -> str:
        text = SunTgBot.get_prompt(intention) + inputText
        tokens = self.__tokenizer(text, return_tensors='pt').to(self.__device)
        size = tokens['input_ids'].shape[1]

        output = self.__model.generate(
            **tokens,
            do_sample=False,
            max_length=size + 375,
            early_stopping=True,
            repetition_penalty=8.,  # 8.
            temperature=1,  # 0.8
            num_beams=10,
        )

        decoded = self.__tokenizer.decode(output[0])
        result = decoded[len(text):]
        print(f'health: {result}')
        if result:
            if result.find(SunTgBot.get_postfix(intention)[0]) >= 0:
                result = result[:result.find(SunTgBot.get_postfix(intention)[0])]
            elif result.find('<s>') >= 0:
                # возвращаем одну из строк, разделенную <s>
                result = random.sample(result.split(sep='<s>')[:-1], k=1)[0]
            else:
                result = result + '...'
        else:
            result = '\nхм... пожалуй здесь я тебе не помогу, разве что пожелаю '
        return inputText + result + SunTgBot.get_postfix(intention)[1]

    @staticmethod
    def get_geo(referent):
        if referent.type == 'государство':
            name_list = []
            for slot in referent.slots:
                if slot.key == 'NAME':
                    name_list.append(slot.value)
            result = min(name_list, key=len)
        else:
            result = referent.name
        return result, referent.type

    def generate_text_weather(self, inputText) -> str:
        res = self.__NER(inputText.upper())
        # print(f'Определяем город: {res}')
        if res.matches:
            geo = self.get_geo(res.matches[0].referent)
        else:
            result = random.sample(['не могу понять в каком месте нужно определить погоду',
                                    'попробуй сказать по другому, например, погода в москве',
                                    'возможно не смог понять в каком месте ты интересуешься погодой'], k=1)[0]
            return result

        print(f'geo: {geo}')
        geo_place = self.__morph.parse(geo[1])[0]
        geo_place_loct = geo_place.inflect({'loct'}).word

        self.__weatherParams['q'] = geo[0].lower()
        response = requests.get(self.__weatherUrl, params=self.__weatherParams)
        print(response)
        if response.status_code == 200:
            result = format_json(response.json(), 'weather', geo_place_loct)
        else:
            result = 'к сожалению я не знаю погоду в этом населенном пункте'
        return result

    def _generate_text_talk(self, user_id, intention) -> str:
        # маркер для входного текста (вопроса)
        inp_mark = '\nx:'
        # маркер для ответа
        out_mark = '\ny:'
        #
        prefix = inp_mark
        for i, t in enumerate(self.history.get(user_id)):
            prefix += t
            prefix += inp_mark if i % 2 == 1 else out_mark
        print(f'prefix: {prefix}')
        tokens = self.__tokenizer(prefix, return_tensors='pt').to(self.__device)
        # tokens = {k: v.to(model.device) for k, v in tokens.items()}
        end_token_id = self.__tokenizer.encode('\n')[0]
        size = tokens['input_ids'].shape[1]
        output = self.__model.generate(
            **tokens,
            eos_token_id=end_token_id,
            do_sample=True,
            max_length=size + 128,
            early_stopping=True,
            repetition_penalty=8.0,  # 3.2,
            temperature=1,
            num_beams=10,
            length_penalty=0.0001,
            pad_token_id=self.__tokenizer.eos_token_id
        )
        decoded = self.__tokenizer.decode(output[0])
        print(f'decoded = {decoded}')
        result = decoded[len(prefix):]
        if result.find(SunTgBot.get_postfix(intention)[0]) >= 0:
            result = result[:result.find(SunTgBot.get_postfix(intention)[0])]
        elif result.find('\n') >= 0:
            # возвращаем строку без \n
            result = result[:result.find('\n')]
        else:
            result = '\nхм...'
        return result

    def generate_text_talk(self, inputText, user_id, intention) -> str:
        self.add_msg_to_history(user_id, inputText)
        result = self._generate_text_talk(user_id, intention)
        self.add_msg_to_history(user_id, result)
        print(self.history)
        return result

    def generate_text_simpletalk(self, inputText, user_id) -> str:
        s_txt = clear_text(inputText)
        self.add_msg_to_history(user_id, inputText)
        # if s_txt in simpleTalk:
        result = random.sample(simpleTalk[s_txt], k=1)[0]
        self.add_msg_to_history(user_id, result)
        print(self.history)
        return result

    def _create_user_history(self, user_id):
        if user_id not in self.history:
            self.history[user_id] = ['Вам известно постановление ученого совета',
                                     'Мне известно, что понедельник начинается в субботу',
                                     'Вы это прекратите',
                                     'Опыт, сын ошибок трудных']
        return

    def add_msg_to_history(self, user_id, msg):
        if self.history.get(user_id):
            self.history.get(user_id).append(msg)
        else:
            self._create_user_history(user_id)
            self.history.get(user_id).append(msg)
        return
