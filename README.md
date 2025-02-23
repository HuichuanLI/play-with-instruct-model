# play-with-instruct-model

从0到1实现一个简单的instruct-gpt(使用facebook/opt-125m)，从训练到模型的构建，主要分成三个步骤：

![image-20250223161313683](./pic/1.png)

- tran_sft.py: 

  - **人工标注：SFT(supervised fine-tune)**

    - instruction -> response数据集，

    - 这一步对于从未经过对齐的GPT3来说比较重要。对于ChatGPT就不必要

    - 如果这里的数据集足够大，忽略后面的RLHF也行。但一般没这么多数据
- train_rm.py：

  - **人类反馈**
    - 基于GPT3SFT对同一instructiong给出的多个response排序
    - 基于排序的pair-wise loss训练response评分模型RM
- tran_rlhf.py：

  - GPT3SFT使用 RM评分模型 作为reward
  - 使用PPO算法进行 fine-tune