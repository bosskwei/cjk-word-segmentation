import torch
import loader
import lstm


def sentence2idx(dictionary, sentence):
    # Tokenize file content
    ids = [dictionary.word2idx[w] for w in sentence]
    return torch.LongTensor(ids)


def space_sentence(sentence, classify):
    assert(len(sentence) == len(classify))
    sentence_spaced = ''
    for w, c in zip(sentence, classify):
        sentence_spaced += w
        if c == 1:
            sentence_spaced += '/'
    return sentence_spaced


def main():
    #
    device = torch.device('cpu')
    corpus = loader.CorpusDense(device)
    #
    sentence = ("从一月一日起，我国再次自主降低关税关税再降意味什么\n"
                "经国务院批准，从2001年1月1日起，我国再次自主降低关税，关税总水平将从目前的16.4％下降为15.3％，平均降幅为6.6％。此次降税共涉及3462个税目，占我国税则税目总数的49％。"
                "近年来，为了适应我国经济发展的需要，以及为了更快融入世界经济潮流，我国政府在降低关税方面作出了积极的努力。其中1996年4月1日，宣布调低4971个税目的进口税率，使我国关税税率的平均水平从35％下降至23％。1997年10月1日，又降低4874个税号商品的进口关税税率，关税平均水平从23％下降到17％。"
                "这次我国政府在加入WTO之前，再次主动大幅下调关税，实现了江泽民主席1996年在菲律宾苏比克湾第四次APEC领导人非正式会议上宣布的“中国将在2000年把关税总水平降低到15％左右”的庄严承诺。它向全世界传递出这样一个信息：我国不但有充分的决心和诚意，而且有足够的信心和能力，致力于发展开放型经济、加强与世界各国和地区的经济贸易合作、努力促进世界经济的发展。"
                "任何国家对本国的产业都有一定程度的保护，其中关税是最主要的保护措施之一。长期以来，我国对进口商品课以高税，在保护国内工业的成长壮大方面有积极作用。但是，在经济逐步走向全球化、科学技术飞速发展的今天，高关税保护政策的弊端逐渐显现，它不利于引进、消化、吸收国外先进的技术设备，不利于企业降低进口成本、参与国际合作和国际竞争。同时，高关税容易刺激一些人铤而走险，走私贩私，破坏正常外贸秩序。"
                "现在，世界投资和贸易自由化趋势日益明显，世界范围的关税壁垒正在降低。据统计，目前发达国家的平均关税已经降到4％以下，发展中国家已经降到13％以下。我国虽然连续多次降低关税，但此次降税之前，关税总水平仍高于发展中国家的平均水平。因此，主动降税在很大意义上是与国际规范接轨，是积极适应WTO对发展中国家的要求，同时与我国经济发展水平也是相适应的。"
                "当然，关税的降低，会使更多国外商品以较低价格进入国内市场，对国内企业产生冲击。但是，冲击同时也是机遇，在我国即将加入WTO，企业即将面临更加严峻的考验之前，我国再次自主降低关税，主动将国内企业推到WTO的门口，无疑给了企业一些应对挑战的提前量。这对企业适应充分竞争环境、增强竞争力，自然是有益处的。")
    sentence_tensor = sentence2idx(corpus.dictionary, sentence)
    assert(len(sentence) == len(sentence_tensor))
    #
    model = torch.load(open('model/256-256x2/model-10-[0.94,0.96].pt', 'rb')).to(device)
    with torch.no_grad():
        model.eval()
        score, _ = model(sentence_tensor.view(-1, 1))
        score = score.view([-1, corpus.segment_num()])
        classify = score.argmax(dim=1).tolist()
        result = space_sentence(sentence, classify)
        print(result)


if __name__ == "__main__":
    main()
