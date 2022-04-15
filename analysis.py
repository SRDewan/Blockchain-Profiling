import json
import sys
import enum
from tqdm import tqdm

weights = {
        "bal_score": 1,
        "trans_score": 2,
        "avg_trans_score": 4,
        "platform_trans_score": 6,
        "platform_avg_score": 8,
        "other_score": 1,
        "defi_user_score": 2,
        "defi_trans_score": 3,
        "nft_user_score": 2,
        "nft_trans_score": 3,
        "token_score": 4,
        "nft_platforms_score": 4,
        }

weights_sum = sum(list(weights.values()))

class Data(enum.Enum):
    PLATFORM_TRANSACTION_COUNT = "Platform Transaction Count"
    USER_COUNTS = "User Counts"
    OTHER_PLATFORM_COUNTS = "Other Platforms Count"
    TOKEN_TRANSFERS = "Token Transfers"
    NFT_TRANSFERS = "NFT Transfers"
    PLATFORM_TRANSACTION_WEIGHT = "10"
    TOKEN_WEIGHT = "8"
    OTHER_PLATFORM_WEIGHT = "5"
    NFT_WEIGHT = "6"


def load_file(file_path):
    ''' Json File Loader '''
    f = open(file_path)
    json_obj = json.load(f)
    return json_obj

def dump_file(data, file_path):
    json_dump = json.dumps(data, indent=5, ensure_ascii=False)

    with open(file_path, "w") as outfile:
        outfile.write(json_dump)

def dict_to_map(mdict):
    data = {}
    ## value can be used later if we decide to give weights to how many times?
    for key, value in mdict.items():
        key = key.lower()
        if key in data:
            data[key] += 1
        else:
            data[key] = 1
    
    return data

def count_overlap_score(dict1, dict2): 
    data1 = dict_to_map(dict1)
    data2 = dict_to_map(dict2)
    
    overlap_count = 0
    total_count = len(data1) + len(data2)
    for key, item in data1.items():
        if key in data2:
            overlap_count += 1
    
    total_count -= overlap_count
    if total_count == 0:
        return 0

    return (overlap_count / total_count) 

def get_similarity(val1, val2):
    if (val1 + val2) == 0:
        return 1
    return 1 - (abs(val1 - val2) / (val1 + val2))

def get_arr_similarity(arr1, arr2):
    score = 0
    tot = 0

    for key in arr1:
        sim = 0
        wt = arr1[key]
        if key in arr2:
            sim = get_similarity(arr1[key], arr2[key])
            wt += arr2[key]

        score += wt * sim
        tot += wt

    for key in arr2:
        sim = 0
        wt = arr2[key]
        if key in arr1:
            continue

        score += wt * sim
        tot += wt

    if tot == 0:
        return 1
    score /= tot
    return score

def get_user_score(d1, d2):
    return [get_similarity(d1["DEFI"]["User Count"], d2["DEFI"]["User Count"]), 
            get_similarity(d1["DEFI"]["Transaction Count"], d2["DEFI"]["Transaction Count"]), 
            get_similarity(d1["NFT"]["User Count"], d2["NFT"]["User Count"]), 
            get_similarity(d1["NFT"]["Transaction Count"], d2["NFT"]["Transaction Count"])]

def compute_matching_score(profile1, profile2):
    feat_scores = {} 
    feat_scores["bal_score"] = get_similarity(profile1["Ether Balance"], profile2["Ether Balance"])
    feat_scores["trans_score"] = get_similarity(profile1["Overall Transaction Count"], profile2["Overall Transaction Count"])
    feat_scores["avg_trans_score"] = get_similarity(profile1["Average Transactions Per Day"], profile2["Average Transactions Per Day"])

    feat_scores["platform_trans_score"] = get_arr_similarity(profile1["Platform Transaction Count"], profile2["Platform Transaction Count"])
    feat_scores["platform_avg_score"] = get_arr_similarity(profile1["Platform Average Transactions Per Day"], profile2["Platform Average Transactions Per Day"])
    feat_scores["other_score"] = get_arr_similarity(profile1["Other Platforms Count"], profile2["Other Platforms Count"])
    [feat_scores["defi_user_score"], feat_scores["defi_trans_score"], feat_scores["nft_user_score"], feat_scores["nft_trans_score"]]= get_user_score(profile1["User Counts"], profile2["User Counts"])

    feat_scores["token_score"] = count_overlap_score(profile1[Data.TOKEN_TRANSFERS.value], profile2[Data.TOKEN_TRANSFERS.value])
    feat_scores["nft_platforms_score"] = count_overlap_score(profile1[Data.NFT_TRANSFERS.value], profile2[Data.NFT_TRANSFERS.value])

    final_matching_score = 0
    for key in feat_scores:
        final_matching_score += weights[key] * feat_scores[key]
    # final_matching_score = platform_score * int(Data.PLATFORM_TRANSACTION_WEIGHT.value) + token_score * int(Data.TOKEN_WEIGHT.value) + other_platforms_score * int(Data.OTHER_PLATFORM_WEIGHT.value) + nft_platforms_score * int(Data.NFT_WEIGHT.value)
    final_matching_score *= 100 / weights_sum
    return final_matching_score

def run_inference(data):
    count = 0
    scores = {}
    ctr = 0

    for user_id1, user_data1 in tqdm(data.items()):
        if ctr > 1000:
            break
        ctr += 1
        # print(user_id, user_data, "\n\n")
        for user_id2, user_data2 in data.items():
            if user_id1 != user_id2:
                key1 = user_id1+"_"+user_id2
                key2 = user_id2+"_"+user_id1

                if key1 not in scores and key2 not in scores:
                    count += 1
                    match_score = compute_matching_score(user_data1, user_data2)
                    scores[key1] = match_score
                    # print("U1=", user_id1, "U2=", user_id2, "Matching_score=", match_score)
    
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))
    dump_file(scores, "score_file.json")
    # print(scores)

def main():
    file_path = sys.argv[1]
    json_obj = load_file(file_path=file_path)
    run_inference(json_obj)

main()
