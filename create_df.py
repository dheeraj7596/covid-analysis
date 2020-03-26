import json
import os
import pandas as pd
import pickle


def get_name(auth_dict):
    mid_name = " ".join(auth_dict["middle"])
    if len(mid_name) == 0:
        return auth_dict["first"] + " " + auth_dict["last"]
    else:
        return auth_dict["first"] + " " + mid_name + " " + auth_dict["last"]


def get_authors(authors_list):
    auth_list = []
    for auth_dict in authors_list:
        name = get_name(auth_dict).lower()
        auth_list.append(name)
    return auth_list


def get_text(text_list):
    body_text = ""
    for body in text_list:
        if len(body_text) == 0:
            body_text = body["text"]
        else:
            body_text = body_text + " " + body["text"]
    return body_text


if __name__ == "__main__":
    base_path = "./data/"
    json_path = base_path + "jsons/"
    final_dict = {}
    cols = ["title", "authors", "body", "abstract"]
    for col in cols:
        final_dict[col] = []

    except_counter = 0
    for filename in os.listdir(json_path):
        if filename.endswith(".json"):
            with open(json_path + filename) as json_file:
                data = json.load(json_file)
                try:
                    title = data["metadata"]["title"]
                    assert len(title) > 0

                    authors = get_authors(data["metadata"]["authors"])
                    assert len(authors) > 0

                    body = get_text(data["body_text"])
                    assert len(body) > 0

                    abstract = get_text(data["abstract"])
                    assert len(abstract) > 0

                    final_dict["title"].append(title.lower())
                    final_dict["authors"].append(authors)
                    final_dict["body"].append(body.lower())
                    final_dict["abstract"].append(abstract.lower())
                except:
                    except_counter += 1
        else:
            continue

    print(except_counter)
    final_df = pd.DataFrame.from_dict(final_dict)
    pickle.dump(final_df, open(base_path + "df_29k.pkl", "wb"))
