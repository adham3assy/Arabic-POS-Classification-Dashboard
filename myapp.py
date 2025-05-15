import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForTokenClassification
import numpy as np
import dash_bootstrap_components as dbc

# Load tokenizer and model
model_path = 'my_arabic_pos_model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = TFAutoModelForTokenClassification.from_pretrained(model_path)

label_mapping = {
    'ADJ': 'صفة', 'ADP': 'حرف جر', 'ADV': 'ظرف', 'AUX': 'فعل مساعد',
    'CCONJ': 'أداة ربط', 'DET': 'أداة تعريف', 'INTJ': 'تعجب', 'NOUN': 'اسم',
    'NUM': 'عدد', 'PART': 'أداة', 'PRON': 'ضمير', 'PROPN': 'اسم علم',
    'PUNCT': 'علامة ترقيم', 'SCONJ': 'أداة ربط', 'SYM': 'رمز', 'VERB': 'فعل', 'X': 'مجهول'
}

# Colors for POS tags
tag_colors = {
    "NOUN": "#636EFA", "VERB": "#EF553B", "ADJ": "#00CC96", "ADV": "#AB63FA",
    "PRON": "#FFA15A", "DET": "#19D3F3", "ADP": "#FF6692", "PUNCT": "#B6E880",
    "CCONJ": "#FF97FF", "NUM": "#FECB52", "X": "#A6CEE3", "PART": "#FDBF6F", 
    "INTJ": "#CAB2D6", "PROPN": "#B2DF8A", "SYM": "#FB9A99", "SCONJ": "#E31A1C",
    "O": "#CCCCCC",
    "صفة": "#00CC96", "حرف جر": "#FF6692", "ظرف": "#AB63FA", "فعل مساعد": "#19D3F3",
    "أداة ربط": "#FF97FF", "أداة تعريف": "#19D3F3", "تعجب": "#CAB2D6", "اسم": "#636EFA",
    "عدد": "#FECB52", "أداة": "#FDBF6F", "ضمير": "#FFA15A", "اسم علم": "#B2DF8A",
    "علامة ترقيم": "#B6E880", "رمز": "#FB9A99", "فعل": "#EF553B", "مجهول": "#A6CEE3"
}

def generate_arabic_overlay():
    arabic_chars = list("ابتثجحخدذرزسشصضطظعغفقكلمنهويءآأؤإئىة")
    overlay = []
    for _ in range(100):
        char = random.choice(arabic_chars)
        top = random.randint(0, 100)
        left = random.randint(0, 100)
        font_size = random.randint(20, 60)
        overlay.append(
            html.Div(
                char,
                style={
                    "position": "absolute",
                    "top": f"{top}%",
                    "left": f"{left}%",
                    "transform": "translate(-50%, -50%)",
                    "fontSize": f"{font_size}px",
                    "color": "rgba(255, 255, 255, 0.1)",
                    "zIndex": "0",
                    "pointerEvents": "none",
                    "userSelect": "none"
                }
            )
        )
    return overlay

# App setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = html.Div([
    html.Div(generate_arabic_overlay()),
    html.Div([
        html.H1("لوحة تحليل الكلمات العربية (POS Tagging)", 
                style={
                    "textAlign": "center", 
                    "color": "white", 
                    "fontWeight": "bold",
                    "fontSize": "36px",
                    "marginBottom": "30px",
                    "paddingTop": "20px"
                }),

        dcc.Textarea(
            id='input-text',
            placeholder='اكتب النص هنا...',
            style={
                'width': '100%', 
                'height': 150, 
                'backgroundColor': '#222', 
                'color': 'white', 
                'fontSize': '18px',
                'direction': 'rtl',
                'border': '1px solid #444',
                'padding': '10px',
                'marginBottom': '20px'
            }
        ),

        html.Button('حلّل النص', id='analyze-btn', n_clicks=0, 
                    style={
                        'width': '100%', 
                        'fontSize': '20px', 
                        'marginBottom': '20px',
                        'backgroundColor': '#8B0000', 
                        'color': 'white', 
                        'border': 'none',
                        'padding': '15px 20px', 
                        'borderRadius': '5px', 
                        'textAlign': 'center',
                        'cursor': 'pointer',
                        'fontWeight': 'bold'
                    }),

        html.Div(id='output-prediction', 
                 style={
                     'textAlign': 'center', 
                     'fontSize': '24px', 
                     'marginBottom': '20px',
                     'direction': 'rtl'
                 }),
        
        html.Div(id='detailed-tags', 
                 style={
                     'textAlign': 'center', 
                     'fontSize': '18px',
                     'direction': 'rtl',
                     'marginBottom': '30px'
                 }),

        dcc.Graph(id='pos-barplot', 
                  style={
                      'display': 'none', 
                      'backgroundColor': '#111', 
                      'borderRadius': '10px',
                      'padding': '20px',
                      'marginBottom': '30px'
                  })
    ], 
    style={
        'position': 'relative', 
        'zIndex': 1, 
        'padding': '30px',
        'maxWidth': '1200px',
        'margin': '0 auto'
    })
], 
style={
    'backgroundColor': '#000', 
    'minHeight': '100vh', 
    'position': 'relative',
    'overflow': 'hidden'
})

def predict_pos(text):
    # Tokenize the text without return_offsets_mapping
    tokens = tokenizer(text, return_tensors="tf", truncation=True)
    
    # Get model predictions
    outputs = model(**tokens)
    preds = tf.argmax(outputs.logits, axis=-1).numpy()[0]
    
    # Get word positions in original text
    word_ids = tokens.word_ids()
    words = tokenizer.convert_ids_to_tokens(tokens["input_ids"].numpy()[0])
    
    # Group predictions by word
    word_to_tag = {}
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue  # Skip special tokens
        word_to_tag.setdefault(word_id, []).append(preds[idx])
    
    # Reconstruct original words with their tags
    results = []
    current_word = ""
    current_word_id = None
    for idx, (word_id, token) in enumerate(zip(word_ids, words)):
        if word_id is None:
            continue  # Skip special tokens
        
        if word_id != current_word_id:
            if current_word:
                # Get the most common tag for the current word
                most_common_tag = max(set(word_to_tag[current_word_id]), 
                                    key=word_to_tag[current_word_id].count)
                label = model.config.id2label[most_common_tag]
                results.append((current_word, label_mapping.get(label, label)))
            current_word = token
            current_word_id = word_id
        else:
            current_word += token.replace("##", "")
    
    # Add the last word
    if current_word:
        most_common_tag = max(set(word_to_tag[current_word_id]), 
                            key=word_to_tag[current_word_id].count)
        label = model.config.id2label[most_common_tag]
        results.append((current_word, label_mapping.get(label, label)))
    
    return results

@app.callback(
    [Output('output-prediction', 'children'),
     Output('detailed-tags', 'children'),
     Output('pos-barplot', 'figure'),
     Output('pos-barplot', 'style')],
    [Input('analyze-btn', 'n_clicks')],
    [State('input-text', 'value')]
)
def update_output(n_clicks, input_text):
    if n_clicks == 0 or not input_text:
        return "", "", go.Figure(), {'display': 'none'}

    preds = predict_pos(input_text)

    # Create word-tag display with tags below
    word_tag_display = html.Div([
        html.Div([
            html.Div(word, style={
                "color": "white",
                "padding": "6px 10px",
                "margin": "4px",
                "display": "inline-block",
                "textAlign": "center",
            }),
            html.Div(tag, style={
                "color": tag_colors.get(tag, "#ccc"),
                "fontSize": "16px",
                "marginTop": "4px",
                "fontWeight": "bold"
            })
        ], style={
            "display": "inline-block", 
            "textAlign": "center", 
            "margin": "6px",
            "verticalAlign": "top"
        })
        for word, tag in preds
    ], style={"direction": "rtl", "textAlign": "center", "paddingBottom": "20px"})

    # Create detailed tags display
    detailed_tags = html.Div([
        html.Span([
            html.Span(word, style={"color": "white"}),
            html.Span(f" [{tag}]", style={
                "color": tag_colors.get(tag, "#ccc"),
                "marginLeft": "5px",
                "fontWeight": "bold"
            })
        ], style={"margin": "0 8px"})
        for word, tag in preds
    ], style={"textAlign": "center", "direction": "rtl"})

    # Count tags for visualization
    tag_counts = {}
    for _, tag in preds:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Create figure with subplots
    fig = make_subplots(
        rows=1, 
        cols=2, 
        specs=[[{"type": "bar"}, {"type": "domain"}]],
        subplot_titles=("توزيع الأنواع", "نسبة كل نوع"),
        horizontal_spacing=0.1
    )

    # Add bar plot
    fig.add_trace(
        go.Bar(
            x=list(tag_counts.values()),
            y=list(tag_counts.keys()),
            orientation='h',
            marker_color=[tag_colors.get(tag, "#ccc") for tag in tag_counts.keys()],
            text=list(tag_counts.values()),
            textposition='auto'
        ),
        row=1, 
        col=1
    )

    # Add donut chart
    fig.add_trace(
        go.Pie(
            labels=list(tag_counts.keys()),
            values=list(tag_counts.values()),
            marker_colors=[tag_colors.get(tag, "#ccc") for tag in tag_counts.keys()],
            hole=0.5,
            textinfo='label+percent',
            insidetextorientation='radial'
        ),
        row=1, 
        col=2
    )

    # Update layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#111',
        plot_bgcolor='#111',
        font_color='white',
        showlegend=False,
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Arial"
        )
    )
    
    # Update bar plot subplot
    fig.update_xaxes(title_text="العدد", row=1, col=1)
    fig.update_yaxes(title_text="النوع", row=1, col=1)
    
    # Update pie chart subplot
    fig.update_traces(
        hoverinfo='label+percent', 
        textfont_size=14,
        row=1, 
        col=2
    )

    return word_tag_display, detailed_tags, fig, {'display': 'block'}

if __name__ == '__main__':
    app.run(debug=True)