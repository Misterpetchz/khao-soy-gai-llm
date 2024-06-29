from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from bert_score import score

def evaluate_bleu(reference, candidate):
    return sentence_bleu([reference.split()], candidate.split())

def evaluate_rouge(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores

def evaluate_bertscore(reference, candidate):
    P, R, F1 = score([candidate], [reference], lang="th")
    return {"precision": P.mean().item(), "recall": R.mean().item(), "f1": F1.mean().item()}

reference = "ร้านอาหาร: ข้าวซอยนิมมาน พิกัด: ซอยนิมมานเหมินท์ 17 รีวิว: ร้านอาหารบรรยากาศดี มีเมนูข้าวซอยและแกงฮังเลที่อร่อยมาก เหมาะสำหรับครอบครัว ราคา: ราคาเริ่มต้นที่ 100 บาท"
candidate = "ร้านอาหาร: ข้าวซอยเสมือน พิกัด: ซอยสุขเกษม 12 รีวิว: ร้านอาหารบรรยากาศดี มีเมนูข้าวซอยและแกงฮังเลที่อร่อยมาก เหมาะสำหรับครอบครัว ราคา: ราคาเริ่มต้นที่ 120 บาท"

bleu_score = evaluate_bleu(reference, candidate)
rouge_score = evaluate_rouge(reference, candidate)
bertscore = evaluate_bertscore(reference, candidate)

print(f"BLEU Score: {bleu_score}")
print(f"ROUGE Score: {rouge_score}")
print(f"BERTScore: {bertscore}")
