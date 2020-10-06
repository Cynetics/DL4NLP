import pandas as pd
# make sure you rename the first column to "lang", i have yet to automatically do this
df_lstm_bpe_clean = pd.read_csv('./gpt-2-LI/classification_report_gpt.csv', delimiter=',')

lang_dict = {}
fam_dict = {}
alphabet_dict = {}

with open('./gpt-2-LI/data/wili-2018/labels.csv', 'r') as f:
    for line in f.readlines():
        s = line.split(';')
        code = s[0]
        lang_dict[code] = s[1]
        fam_dict[code] = s[5]
        alphabet_dict[code] = s[6]


def analyze_df(df):
    df['language'] = df['lang'].map(lang_dict)
    df['fam'] = df['lang'].map(fam_dict)
    df['alphabet'] = df['lang'].map(alphabet_dict)

    #with pd.option_context('display.max_rows', None):
    lowest_prec = df.sort_values('precision')[['language', 'precision' ]][:40]
    lowest_rec = df.sort_values('recall')[['language', 'recall']][:40]
    lowest_f1 = df.sort_values('f1-score')[['language', 'f1-score' ]][:40]

    lowest_prec.plot.bar()
    print()

analyze_df(df_lstm_bpe_clean)
