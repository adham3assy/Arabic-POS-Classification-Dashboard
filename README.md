# 🧠 Arabic POS Tagging Dashboard

This project is an interactive web dashboard for **Part-of-Speech (POS) tagging** of Arabic text, built using **Dash by Plotly** and powered by a **transformer-based NLP model** from HuggingFace's Transformers library.

It allows users to input Arabic sentences and visualize the predicted POS tags, including colored annotations and tag distribution charts, making it a powerful educational and analytical tool for understanding Arabic grammar structure.

---

## 🌟 Features

- ✍️ **Real-time POS tagging** of Arabic input text  
- 🎨 **Dynamic visual output** with colored tokens and tag overlays  
- 📊 **Interactive visualizations**: bar chart and donut chart of POS distribution  
- 🌙 **Fully dark-themed UI** with decorative Arabic character background  
- 🌐 **Right-to-left (RTL)** layout optimized for Arabic content  
- 🧠 **Transformer-based deep learning model** for high-quality linguistic predictions  

---

## ⚙️ Tech Stack

### 🧪 NLP & Machine Learning
- 🤗 [Hugging Face Transformers](https://huggingface.co/transformers/)
- 🤖 `TFAutoModelForTokenClassification` with a fine-tuned BERT-based model
- 📝 `AutoTokenizer` for tokenization and word alignment

### 💻 Web App & Visualization
- 📊 `Dash` and `Dash Bootstrap Components` for responsive UI
- 🧩 `Plotly Express` and `Graph Objects` for interactive visualizations
- 🌑 Layout for a modern dark theme with Arabic support

---

## 🧠 Pretrained Model

This dashboard uses a fine-tuned **transformer model for POS tagging** of Arabic text. You can load your own or use one from Hugging Face. The app expects the model in the following format:

- A saved directory with:
  - `config.json`
  - `tf_model.h5`
  - `tokenizer_config.json`
  - `vocab.txt` / `tokenizer.json`

Replace the model path:
```python
model_path = 'my_arabic_pos_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TFAutoModelForTokenClassification.from_pretrained(model_path)
