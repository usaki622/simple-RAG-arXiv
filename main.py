# ============== 配置部分 ===================
import json
import logging
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info("启动日志")

# 1.配置好大模型API
from langchain_openai import *
import os

os.environ["OPENAI_API_KEY"] = "None"
os.environ["OPENAI_API_BASE"] = "http://10.58.0.2:8000/v1"
llm_completion = ChatOpenAI(model_name="Qwen2.5-14B") # 原先的API太老了，此处用新的
# llm_chat = OpenAIChat(model_name="Qwen2.5-14B") # 不需要多轮对话跟踪上下文，就不引用这个了

# 2.配置好嵌入模型，连接到数据库
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
db = Milvus(embedding_function=embedding, collection_name="arXiv_Back",connection_args={"host": "10.58.0.2", "port": "19530"})

# ============== 正式部分 ===================

# 1.导入问题
questions=[]
def getQuestions(questions_file):
    try:
        logging.info(f"尝试读取问题列表: {questions_file}")
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_json = json.load(f)
        logging.info(f"成功加载文件: {questions_file}")
    except FileNotFoundError:
        logging.error(f"文件未找到: {questions_file}")
    except json.JSONDecodeError as e:
        logging.error(f"JSON 解码错误: {e}")
    except Exception as e:
        logging.error(f"加载文件时发生未知错误: {e}")

    for item in questions_json:
        question = item.get('question', '')
        if not question:
            logging.warning(f"跳过: {item}")
            continue

        questions.append(question)

# 2.从问题列表中提取到关键词
def getKeyWord(question):
    question_prompt = f"请提取出下面问题中最重要的一个关键词，要求务必用英文，绝对不能含有任何中文，并且务必只有一个词 {question}"
    response = llm_completion(question_prompt)
    # print(response.content)
    return response.content

# 3.从数据库中寻找相关文献
docs=[]
sources=[]
def getDocs(questions):
    for i in range(len(questions)):
        # 用户给出的问题或陈述不一定能够匹配向量数据库的查询，因此需要反复尝试
        cnt = 0
        while cnt < 3:
            keyword = getKeyWord(questions[i])
            best_doc=db.similarity_search(keyword)

            # print(best_doc)

            random_index = random.randint(0, len(best_doc) - 1)

            if not best_doc:
                cnt += 1
                logger.info(f"第{cnt}次尝试：没有找到相关文档，正在重新优化查询。")
            else:
                sources.append(f"https://arxiv.org/abs/{best_doc[random_index].metadata['access_id']}")
                docs.append(best_doc[random_index].page_content)
                break
        # 反复查询都没有查到
        if cnt == 3:
            sources.append("None")
            docs.append("None")
    # print(docs)

# 4.结合文献给出回答
answers=[]
def getAnswers(doc, question):
    question_prompt = f"请根据以下内容：{doc[:200]}，回答问题：{question}，结果返回中文"

    if(doc == "None"):
        answers.append("抱歉，未查询到您当前问题相关的资料，请尝试其它问题！")
    else:
        response = llm_completion(question_prompt)
        # print("这条的结果是：")
        # print(response.content)
        answers.append(response.content)

# 5.写入到输出文件里
def outputAnswers(output_file):
    data = [{"question": q, "answer": a,"source": s} for q, a, s in zip(questions, answers, sources)]

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logging.info(f"数据已成功写入到 {output_file}")
    except Exception as e:
        logging.error(f"写入文件时发生错误: {e}")

# 总的方法，main入口中执行这个就可以了
def answer(input_file, output_file):
    # 第一步
    getQuestions(input_file)
    # 第二步
    getDocs(questions)
    # 第四步
    for i in range(len(questions)):
        getAnswers(docs[i], questions[i])
    # 第五步
    outputAnswers(output_file)


if __name__ == '__main__':
    questions_file = "./questions.jsonl"
    output_file='answer.json'
    answer(questions_file,output_file)