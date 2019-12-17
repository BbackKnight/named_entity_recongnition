from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from data import build_corpus
from evaluating import Metrics
from evaluate import ensemble_evaluate

BiLSTMCRF_MODEL_PATH = './ckpts/bilstm_crf_test.pkl'

REMOVE_O = False  # 在评估的时候是否去除O标记


def main():
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    dev_word_lists, dev_tag_lists = build_corpus("dev", make_vocab=False)
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)

    word_string = "傅城州，博士，广东药科大学医药信息工程学院教师，中国计算机学会会员，中国计算机学会青年计算机科技论坛（YOCSEF）广州分论坛AC委员、学术秘书。本科和硕士（推免）毕业于华南师范大学计算机学院软件工程专业，2017年6月获得华南师范大学服务计算理论与技术理学博士学位（导师：汤庸教授）。"
    word_list = []
    test_word_lists = []
    for word in word_string:
        word_list.append(word)

    test_word_lists.append(word_list)

    print("加载并评估bilstm+crf模型...")
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id, for_crf=True)
    bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
    bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning
    # test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
    #     test_word_lists, test_tag_lists, test=True
    # )
    print(crf_tag2id)
    id2tag = {v: k for k, v in crf_tag2id.items()}
    lstmcrf_pred = bilstm_model.test(test_word_lists, crf_word2id, crf_tag2id)
    for word_list, tag_list in zip(test_word_lists, lstmcrf_pred):
        for word, tag in zip(word_list, tag_list):
            print(word, " ", tag.item(), " ", id2tag[tag.item()])

    print(lstmcrf_pred)

    # # 将id转化为标注
    # pred_tag_lists = []
    # id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
    # for i, ids in enumerate(lstmcrf_pred):
    #     tag_list = []
    #     for j in range(lengths[i]):  # crf解码过程中，end被舍弃
    #         tag_list.append(id2tag[ids[j].item()])
    #     pred_tag_lists.append(tag_list)


if __name__ == "__main__":
    main()
