from flask import Flask, render_template, request, jsonify
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import pipeline

app = Flask(__name__)

print('Model loading started')
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
print('Model loading done')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        src_lang = request.form['src_lang']
        tgt_lang = request.form['tgt_lang']
        print(text, src_lang, tgt_lang)
        #selected_language = request.form['language']
        translation = translate(text, src_lang, tgt_lang)
        print(translation)
        return jsonify(translation=translation)
    else: return render_template('ui.html', translation=None)

@app.route('/summarize', methods=['GET','POST'])
def summarize_text():
    if request.method == 'POST':
        text = request.form['text']
        summary = summarize(text)
        print(summary[0]['summary_text'])
        s = summary[0]['summary_text']
        return jsonify(summary=s)
    else: return render_template('ui2.html', summary=None)

def summarize(text):
    return summarizer(text, max_length=130, min_length=30, do_sample=False)

def translate(text, src_lang, tgt_lang):
    language_codes = {
        "English": "en_XX",
        "Arabic": "ar_AR",
        "Czech": "cs_CZ",
        "German": "de_DE",
        "Spanish": "es_XX",
        "Estonian": "et_EE",
        "Vietnamese": "vi_VN",
        "Finnish": "fi_FI",
        "French": "fr_XX",
        "Hindi": "hi_IN",
        "Italian": "it_IT",
        "Japanese": "ja_XX",
        "Korean": "ko_KR",
        "Lithuanian": "lt_LT",
        "Nepali": "ne_NP",
        "Russian": "ru_RU",
        "Chinese": "zh_CN",
        "Afrikaans": "af_ZA",
        "Azerbaijani": "az_AZ",
        "Bengali": "bn_IN",
        "Croatian": "hr_HR",
        "Indonesian": "id_ID",
        "Malayalam": "ml_IN",
        "Thai": "th_TH",
        "Tagalog": "tl_XX",
        "Ukrainian": "uk_UA"
    }
    tokenizer.src_lang = language_codes[src_lang]
    translation = tokenizer.batch_decode(
        model.generate(**tokenizer(text, return_tensors="pt"), forced_bos_token_id=tokenizer.lang_code_to_id[language_codes[tgt_lang]]),
        skip_special_tokens=True)
    print(translation[0])
    return translation[0]

if __name__ == '__main__':
    app.run(debug=True)