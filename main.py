import re
import random
import ollama
from datasets import load_dataset
from deep_translator import GoogleTranslator

MODEL = "llama3.2:1b"
DATASET = "LVSTCK/nq_open-mk"
FASTTEXT_LID_PATH = "lid.176.bin"

TEMPERATURES = [0.0]
TOP_PS = [0.3]

NUM_TRIALS = 5
NUM_FEWSHOT = 3
MAX_TOKENS = 80

SEED = 42
random.seed(SEED)

translator = GoogleTranslator(source="mk", target="en")


def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()


def load_fasttext_model(path):
    import fasttext
    return fasttext.load_model(path)


# procenuvanje koj jazik e so fasttest lid
def fasttext_lang(ft, text):
    text = normalize_spaces(text)

    labels, scores = ft.predict(text, k=1)
    if not labels:
        return "unknown", 0.0
    return labels[0].replace("__label__", ""), float(scores[0])


# proverka na tekstot dali e makedonski spored povekje kriteriumi
def is_clean_mk(ft, text):
    lang, score = fasttext_lang(ft, text)
    ok = (lang == "mk") and (score >= 0.45)
    return ok, lang, score


# randomised loadiranje od dataset
def load_qa_pairs_split(ds_split, n):
    idxs = list(range(len(ds_split)))
    random.shuffle(idxs)

    pairs = []
    for i in idxs:
        q = (ds_split[i].get("question") or "").strip()
        answers = ds_split[i].get("answer") or []
        a = answers[0].strip() if answers else ""

        if q:
            pairs.append((q, a))

        if len(pairs) >= n:
            break

    return pairs


def load_fewshot_and_test(num_fewshot, num_trials):
    """
    Load few-shot examples from train, and test pool from test.
    If no test split exists, fallback to train (but then we avoid overlap by sampling disjoint sets).
    """
    ds = load_dataset(DATASET)

    train_split = ds["train"]

    if "test" in ds:
        test_split = ds["test"]
        fewshot_examples = load_qa_pairs_split(train_split, num_fewshot)
        test_pool = load_qa_pairs_split(test_split, num_trials)
        return fewshot_examples, test_pool

    # Fallback: no test split available.
    # Sample disjointly from train: first few-shot then test from remaining.
    all_train = load_qa_pairs_split(train_split, num_fewshot + num_trials)
    fewshot_examples = all_train[:num_fewshot]
    test_pool = all_train[num_fewshot:num_fewshot + num_trials]
    return fewshot_examples, test_pool


def mix_text(text: str, how_many: int = 2):
    mk_tokens = text.split()
    en_tokens = translator.translate(text).split()

    if not mk_tokens or not en_tokens:
        return text

    n = min(how_many, len(mk_tokens), len(en_tokens))
    idxs = list(range(min(len(mk_tokens), len(en_tokens))))
    random.shuffle(idxs)
    chosen = idxs[:n]

    for i in chosen:
        mk_tokens[i] = en_tokens[i]
    return " ".join(mk_tokens)


# prompts
def system_prompt():
    return (
        "Ти си корисен асистент кој одговара САМО на македонски јазик.\n"
        "Смееш да користиш САМО македонска кирилица.\n"
        "Не смееш да користиш латиница или англиски зборови.\n"
        "Одговори со ЕДНА јасна реченица."
    )


def prompt_baseline(q):
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": f"Прашање: {q}\nОдговори на македонски (кирилица) со една реченица."}
    ]


def prompt_fewshot(q, examples):
    msgs = [{"role": "system", "content": system_prompt()}]
    for ex_q, ex_a in examples:
        msgs.append({"role": "user", "content": f"Прањаше: {ex_q}"})
        msgs.append({"role": "assistant", "content": f"{ex_a}"})
    msgs.append({"role": "user", "content": f"Прашање: {q}\nОдговори со една јасна реченица."})
    return msgs


def prompt_icm(q: str, examples, how_many: int = 2):
    msgs = [{"role": "system", "content": system_prompt()}, {
        "role": "user",
        "content": (
            "Примерите подолу содржат мешани зборови, но твојот одговор МОРА да биде чисто на македонски (кирилица), без латиница и без англиски зборови. Одговори со една јасна реченица."
        )
    }]

    for ex_q, ex_a in examples:
        mixed_q = mix_text(ex_q, how_many=how_many)
        mixed_a = mix_text(ex_a, how_many=how_many)
        msgs.append({"role": "user", "content": f"Прашање: {mixed_q}"})
        msgs.append({"role": "assistant", "content": f"{mixed_a}"})

    msgs.append({"role": "user", "content": f"Прашање: {q}\nОдговори со една јасна реченица."})
    return msgs


def prompt_echo():
    s = "Денес е убав ден и сакам да зборувам на македонски."
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": "Повтори ја реченицата ТОЧНО:\n" + s}
    ]


# chat so Ollama
def ask_ollama(messages, temperature, top_p):
    resp = ollama.chat(
        model=MODEL,
        messages=messages,
        options={"temperature": temperature, "top_p": top_p, "num_                                                                                                                                                                                  predict": MAX_TOKENS}
    )
    return resp["message"]["content"]


def run():
    print("Loading data...")
    print("Loading data...")
    fewshot_examples, test_pool = load_fewshot_and_test(NUM_FEWSHOT, NUM_TRIALS)

    print("Loading fastText...")
    ft = load_fasttext_model(FASTTEXT_LID_PATH)

    # site strategii
    strategies = [
        "baseline",
        "fewshot",
        "icm",
        "echo",
    ]

    for name in strategies:
        for temp in TEMPERATURES:
            for top_p in TOP_PS:
                print(f"strategy={name} temp={temp} top_p={top_p}")

                for i, (q, a) in enumerate(test_pool, 1):
                    if name == "baseline":
                        messages = prompt_baseline(q)
                    elif name == "fewshot":
                        messages = prompt_fewshot(q, fewshot_examples)
                    elif name == "icm":
                        messages = prompt_icm(q, fewshot_examples, how_many=2)
                    else:
                        messages = prompt_echo()

                    out = ask_ollama(messages, temp, top_p)

                    ok, lang, score = is_clean_mk(ft, out)

                    print(f"\n{i}. Q: {q}")
                    print(f"   A: {out}")
                    print(f"   lang={lang} score={score:.2f} PASS={ok}")

                    # if ok:
                    #     print("\nFOUND ONE CLEAN OUTPUT -> stopping.\n")
                    #     return

    print("\nChat cycle finished.")


if __name__ == "__main__":
    run()
