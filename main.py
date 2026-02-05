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
def load_qa_pairs(n):
    ds = load_dataset(DATASET)["train"]

    idxs = list(range(len(ds)))
    random.shuffle(idxs)

    pairs = []
    for i in idxs:
        q = ds[i]["question"].strip()
        answers = ds[i]["answer"]
        a = answers[0].strip() if answers else ""

        pairs.append((q, a))
        if len(pairs) >= n:
            break

    return pairs

translator = GoogleTranslator(source="mk", target="en")

def mix_text(text, how_many=2):
    mk = text.split()
    en = translator.translate(text).split()

    n = min(how_many, len(mk), len(en))
    for i in range(n):
        mk[i] = en[i]
    return " ".join(mk)


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
    block = ""
    for ex_q, ex_a in examples:
        block += f"Прашање: {ex_q}\nОдговор: {ex_a}\n\n"
    user = block + f"Прашање: {q}\nОдговор:"
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user}
    ]


def prompt_icm(q, examples, how_many=2):
    block = ""
    for ex_q, ex_a in examples:
        block += f"Прашање: {mix_text(ex_q, how_many)}\nОдговор: {mix_text(ex_a, how_many)}\n\n"
    user = (
            "Примерите имаат мешани зборови, ама ти одговори чисто на македонски (кирилица):\n\n"
            + block
            + f"Прашање: {q}\nОдговор:"
    )
    return [
        {"role": "system", "content": system_prompt()},
        {"role": "user", "content": user}
    ]


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
        options={"temperature": temperature, "top_p": top_p, "num_predict": MAX_TOKENS}
    )
    return resp["message"]["content"]


def run():
    print("Loading data...")
    qa = load_qa_pairs(NUM_TRIALS + NUM_FEWSHOT)

    fewshot_examples = qa[:NUM_FEWSHOT]
    test_pool = qa[NUM_FEWSHOT:NUM_FEWSHOT + NUM_TRIALS]

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
