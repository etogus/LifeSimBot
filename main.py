import random
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

BOT_CONFIG = {
    'intents': {
        'hello': {
            'examples': ['Привет', 'Добрый день', 'Добрый вечер', 'Здравствуйте', 'Шалом', 'Приветики'],
            'responses': ['Привет, человек!', 'Здравствуйте']
        },
        'bye': {
            'examples': ['Пока', 'Досвидания', 'До свидания', 'Прощайте'],
            'responses': ['Ещё увидимся', 'Если что, я тут']
        }
    },

    'failure_phrases': [
        'Непонятно. Перефразируй, пожалуйста',
        'Я ещё только учусь. Не умею на такое отвечать'
    ]
}

X_texts = []  # реплики
y = []  # их классы

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        X_texts.append(example)
        y.append(intent)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_texts)
clf = LogisticRegression().fit(X, y)


def get_intent(question):
    question_vector = vectorizer.transform([question])
    intent = clf.predict(question_vector)[0]

    examples = BOT_CONFIG['intents'][intent]['examples']
    for example in examples:
        dist = nltk.edit_distance(question, example)
        dist_percentage = dist / len(example)
        if dist_percentage < 0.4:
            return intent


def filter_text(text):
    text = text.lower()
    text = [c for c in text if c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя- ']
    text = ''.join(text)
    return text


def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        phrases = BOT_CONFIG['intents'][intent]['responses']
        return random.choice(phrases)


enc = 'utf-8'

with open('dialogues.txt', encoding=enc) as f:
    content = f.read()

dialogues = [dialogue_line.split('\n') for dialogue_line in content.split('\n\n')]

questions = set()
qa_dataset = []  # [[q, a]]

for replicas in dialogues:
    if len(replicas) < 2:
        continue
    question, answer = replicas[:2]
    question = filter_text(question[2:])
    answer = answer[2:]
    if question and question not in questions:
        questions.add(question)
        qa_dataset.append([question, answer])

qa_by_word_dataset = {}
for question, answer in qa_dataset:
    words = question.split(' ')
    for word in words:
        if word not in qa_by_word_dataset:
            qa_by_word_dataset[word] = []
        qa_by_word_dataset[word].append((question, answer))

qa_by_word_dataset_filtered = {word: qa_list
                               for word, qa_list in qa_by_word_dataset.items()
                               if len(qa_list) < 1000}


def generate_answer_by_text(text):
    text = filter_text(text)
    words = text.split(' ')
    qa = []
    for word in words:
        if word in qa_by_word_dataset_filtered:
            qa += qa_by_word_dataset[word]
    qa = list(set(qa))[:2000]

    results = []
    for question, answer in qa:
        dist = nltk.edit_distance(question, text)
        dist_percentage = dist / len(question)
        results.append([dist_percentage, question, answer])

    if results:
        dist_percentage, question, answer = min(results, key=lambda pair: pair[0])
        if dist_percentage < 0.2:
            return answer


def get_failure_phrase():
    phrases = BOT_CONFIG['failure_phrases']
    return random.choice(phrases)


def bot(question):
    # NLU
    intent = get_intent(question)

    # Получение ответа

    # правила
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            return answer

    # генеративная модель
    answer = generate_answer_by_text(question)
    if answer:
        return answer

    # заглушка
    answer = get_failure_phrase()
    return answer


# def chatWithBot(inputText):
#     currentText = bag_of_words(inputText, words)
#     currentTextArray = [currentText]
#     numpyCurrentText = numpy.array(currentTextArray)
#
#     if numpy.all((numpyCurrentText == 0)):
#         return "I didn't get that, try again"
#
#     result = myChatModel.predict(numpyCurrentText[0:1])
#     result_index = numpy.argmax(result)
#     tag = labels[result_index]
#
#     if result[0][result_index] > 0.7:
#         for tg in data["intents"]:
#             if tg['tag'] == tag:
#                 responses = tg['responses']
#
#         return random.choice(responses)
#
#     else:
#         return "I didn't get that, try again"