import json

K=5
input_file="../data/glove.6B.50d.json"
out_file="../data/glove.6B.{}d.json".format(K)
ori_word_vec = json.load(open(input_file, "r", encoding="utf-8"))

"""
[
{"word": "the", "vec": [0.418, 0.24968]},
{"word": "hello", "vec": [0.418, 0.24968]},
]
"""

all_vec_list = []
for cur_id, word in enumerate(ori_word_vec):
    w = word['word']
    w = w.lower()
    vec = word['vec'][0:K]
    all_vec_list.append({"word":w,"vec":vec})

json.dump(all_vec_list, open(out_file,"w", encoding="utf-8"))
print("outfile:", out_file)



