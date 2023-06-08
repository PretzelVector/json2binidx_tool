# jsonl to binidx tool

This repository is greatly simplified from https://github.com/EleutherAI/gpt-neox, to ONLY convert .jsonl into .bin and .idx , can serve for dataset preparation of RWKV model (see https://github.com/BlinkDL/RWKV-LM), 

## The current RWKV models use GPT Neox tokenizer 20B_tokenizer.json
```
python tools/preprocess_data.py --input ./sample.jsonl --output-prefix ./data/sample --vocab ./20B_tokenizer.json --dataset-impl mmap --tokenizer-type HFTokenizer --append-eod
```

## The multilingual rwkv-4-world models use a new tokenizer rwkv_vocab_v20230424.txt.
```
python tools/preprocess_data.py --input ./sample.jsonl --output-prefix ./data/sample --vocab ./rwkv_vocab_v20230424.txt --dataset-impl mmap --tokenizer-type RWKVTokenizer --append-eod
```

The jsonl format sample (one line for each document):
```json
{"text": "This is the first document."}
{"text": "Hello\nWorld"}
{"text": "1+1=2\n1+2=3\n2+2=4"}
```
generated by code like this:
```python
ss = json.dumps({"meta": meta, "text": text}, ensure_ascii=False)
out.write(ss + "\n")
```

# binidx merge tool
Takes a list of binidx datasets and merges them into a single one.
```bash
python tools/merge_binidx.py --input $(find $input_path -name "*.idx" | sed 's/\.idx//g') --output output_merged --vocab ./20B_tokenizer.json --tokenizer-type HFTokenizer
```
