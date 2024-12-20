# RAG 作业报告

## 一、准备环境

配置好python环境，并准备好相关依赖。

其中需要注意的是，下述依赖需要安装正确的版本：

```python
pip install pymilvus==2.2.6
pip install openai==0.28
pip install protobuf==3.20.0
```

其它依赖直接安装最新版即可。

需要指出的是，moodle上部分API比较陈旧，使用存在一定困难，因此在配置部分使用的是新的API，详情请参照代码和对应API文档部分。



## 二、代码思路

本代码作业要求构建一个针对 arXiv 的知识问答系统。

我们将助教提供的10个问题视为输入（用户只需参照这种模式，创建文件即可视为输入），而后使用大模型和arxiv数据库对每个问题做如下处理：

- 第一步，使用大模型对输入的问题进行prompt优化，提取出其中的关键词；
- 第二步，利用关键词，在arxiv数据库中查找相关的文献；
- 第三步，根据查找到的文献，与大模型再次进行交互，让其进行分析，并给出最后的答案；
- 第四步，将答案输出。



## 三、代码说明

代码请参照main.py，其中加入了详细的注释。需要在校园网环境下运行。



下面重点说明一些基本功能以外的优化部分：

### 优化一

**问题：**用户初始输入可能比较粗糙，无法找到对应文档

**解决方式：**对用户的初始输入使用大模型进行优化润色，提取出里面的关键词，以此提高找到对应文档的概率。

```python
# 2.从问题列表中提取到关键词
def getKeyWord(question):
    question_prompt = f"请提取出下面问题中最重要的一个关键词，要求务必用英文，绝对不能含有任何中文，并且务必只有一个词 {question}"
    response = llm_completion(question_prompt)
    # print(response.content)
    return response.content
```

此处的question_prompt进行过多次调整。

### 优化二

**问题：**用户给出的问题或陈述不一定能够匹配向量数据库的查询。

**解决方式：**可以进行反复尝试，用户给出的问题或陈述若无匹配内容，则返回重新对prompt再次进行关键词提取，多次优化并反复进行。若超过三次，则判定为无法找到，给出单独的输出。

```python
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
            # 后续处理answer时，会对这种情况做单独处理
    # print(docs)
```

### 优化三

**问题：**单次的查询可能无法寻找到用户所期望的答案，需要通过多轮的搜索和尝试才能获得较为准确的答案

**解决方式：**代码同样在上面。根据测试，每次查到的文档不止一个，因此可以每次都随机从中选出一个结果，使得用户在不同轮数的尝试下依据的文档资料不同，从而尽可能获得多种答案。



最终得到的结果请参照文件夹中的answer.json文件，结果如下（省略了中间部分）：

```json
[
    {
        "question": "什么是大语言模型？",
        "answer": "根据您提供的内容，它主要讨论了模型管理和一种基于代数规范的新模型管理方法，并没有直接定义“大语言模型”。不过，我可以为您解释一下什么是大语言模型。\n\n大语言模型是指一类在大规模语料库上训练的语言模型，这类模型通常具有大量的参数（如数十亿甚至更多），能够捕捉语言中的复杂模式。它们通过处理和理解大量的文本数据来学习语言的结构和语义，因此在生成文本、翻译、摘要等多种自然语言处理任务中表现出色。",
        "source": "https://arxiv.org/abs/2301.04846"
    },
    {
        "question": "形式化软件工程是什么？",
        "answer": "形式化软件工程是指在软件开发过程中使用形式化的数学方法来精确描述软件需求、设计和行为的工程方法。这种方法强调使用严格的数学语言和逻辑来定义软件系统的各个方面，从而能够在软件开发的早期阶段就识别并解决潜在的问题。通过形式化的方法，可以提高软件的质量、可靠性和安全性，减少软件错误和漏洞，提高软件的可维护性和可验证性。上述内容提到的框架正是为了支持这种形式化方法的设计和分析而开发的，它提供了一种支持规范语言的工具，以便于软件系统的规范描述。",
        "source": "https://arxiv.org/abs/0910.0747"
    },
...
...
    {
        "question": "人造原子是什么？",
        "answer": "根据您提供的上下文，这里讨论的是原子的结构。通常，物理学家和化学家认为，在没有电场的情况下，原子的原子核位于电子云的中心，原子本身不具有永久的电偶极矩。而“人造原子”这一概念通常指的是通过实验技术创造出来的具有特定性质的原子模型或系统，它们可能与自然存在的原子结构有所不同，例如，通过量子点、冷原子技术等方式构造出的具有特定电子分布的原子系统。这些系统可以用来研究原子物理中的各种现象，或者用于特定的技术应用中。但根据您提供的信息，无法直接得出“人造原子”的具体定义，更详细的信息可能需要特定的上下文。总的来说，“人造原子”是一种人为构造、研究或者技术应用中的原子模型。",
        "source": "https://arxiv.org/abs/1010.2425"
    }
]
```

其中answer代表了用户的提问，answer代表了系统的回答，source代表了依据的来源。

