import streamlit as st
import pandas as pd
import numpy as np

import catboost
import sklearn
import pickle


@st.cache
def load_domspec_data():
  exptypes = [x.strip()   for x in open("exptypes.txt", "r").readlines()]
  prodtypes = [x.strip()   for x in open("prodtypes.txt", "r").readlines()]
  return((exptypes, prodtypes))

exptypes, prodtypes = load_domspec_data()


@st.cache
def load_pretrained_models():
  def cbload(i):
    from_file = catboost.CatBoostClassifier()
    from_file.load_model("catboost_p{}.cbm".format(i))  
    return(from_file)

  cb_models = [cbload(i)   for i in range(10)]
  lr_models = pickle.load(open("logregr_models.pickle", 'rb'))
  cb_thrstats = pickle.load(open("catboost_thr.pickle", 'rb'))
  lr_thrstats = pickle.load(open("logregr_thr.pickle", 'rb'))
  return(cb_models, lr_models, cb_thrstats, lr_thrstats)


def transform_to_input_format(df):
  df2 = df.groupby("EXP_TYPE").apply(lambda x: pd.Series([x.shape[0], 
    x.AMOUNT.sum()], index= ["n", "AMOUNT"]))
  df2.loc[:, "AMOUNT_prop"] = df2.AMOUNT / df2.AMOUNT.sum()
  df2.loc[:, "n_prop"] = df2.n / df2.n.sum()
  df2.loc[:, "AMOUNT_per_n"] = [0.0 if np.isnan(x) else x  
                                for x in df2.AMOUNT / df2.n]
  nc = len(exptypes)
  datax = np.zeros((1, 3*len(exptypes)))
  datax[0, range(nc)] = df2.AMOUNT_prop
  datax[0, range(nc, nc + nc)] = df2.n_prop
  datax[0, range(2*nc, 2*nc + nc)] = df2.AMOUNT_per_n

  return(datax)

  
  
  


cb_models, lr_models, cb_thrstats, lr_thrstats = load_pretrained_models()

st.title('Рекомендательная система для продуктов банка')

with st.expander("Описание формата входной таблицы"):
  st.markdown("""
  В таблице должны присутствовать столбцы **EXP_TYPE** и **AMOUNT**. Столбец
  **AMOUNT** содержит размер транзакции в рублях, **EXP_TYPE** --- категорию 
  продутка, к которму относится транзакция. Возможные значения для **EXP_TYPE**:
   - {}

  В рамках системы рассматриваются следующие возможные продукты банка для 
  рекомендации клиенту:
   - {}
  """.format("\n - ".join(exptypes), "\n - ".join(prodtypes)))

upfile = st.file_uploader('Загрузка таблицы данных транзакций пользователя ' + 
  'для анализа', ["csv", "xlsx"])

if upfile is None:
  st.stop()

if upfile.name.endswith(".csv"):
  df = pd.read_csv(upfile)
elif upfile.name.endswith(".xlsx"):
  df = pd.read_excel(upfile)
df.EXP_TYPE = pd.Categorical(df.EXP_TYPE, exptypes)

with st.expander("Содержимое загруженной таблицы"):
  st.dataframe(df)

@st.cache
def predictions(datax):
  return(pd.DataFrame({
      "cb": [cb_models[i].predict_proba(datax)[0][1] 
             for i in range(len(prodtypes))],
      "lr": [lr_models[i].predict_proba(datax)[0][1] 
             for i in range(len(prodtypes))]
    }))

pr = predictions(transform_to_input_format(df))


targmetr = st.radio("При формировании рекомендаций контролировать метрику",
  ["precision", "recall"])
targmetrval = st.slider("Значение метрики", 0., 1., 0.9)

targmetr = 1 if targmetr == "precision" else 2
metrother = 2 if targmetr == 1 else 1

def find_approp_thresholds(thrstats, target, what): #what=1 -- prec, 2 -- recall
  r = []
  for i in range(len(thrstats[1])):
    l = thrstats[what][i] >= target
    if l.sum() > 0:
      j = np.where(l)[0][np.argmin(thrstats[what][i][l])]
    else:
      j = np.argmax(thrstats[what][i])
    r.append({"thr": thrstats[0][j], "prec": thrstats[1][i][j], 
              "recall": thrstats[2][i][j]})
  return pd.DataFrame(r)
cbt = find_approp_thresholds(cb_thrstats, targmetrval, targmetr)
lrt = find_approp_thresholds(lr_thrstats, targmetrval, targmetr)


r = []
for i in range(pr.shape[0]):
  if cbt.iloc[i, metrother] > lrt.iloc[i, metrother]:
    if pr.cb[i] >= cbt.thr[i]:
      r.append(prodtypes[i])
  else:
    if pr.lr[i] >= lrt.thr[i]:
      r.append(prodtypes[i])

st.write("Продукты, рекомендуемые пользователю:")
if len(r) > 0:
  st.markdown("\n".join([" - {}".format(x) for x in r]))
else:
  st.markdown("*нет рекомендаций с заданными параметрами надёжности критерия*")

st.write("")
st.write("")
st.markdown("*Рекомендации получены на основе результатов CatBoost и Logistic"+
  "Regression моделей (для каждого продукта выбиралась модель с наибольшими"+
  "precision/accuracy). Ниже представлены подробные значения применения "+
  "моделей*")

st.write("CatBoost:")
st.table(pd.DataFrame({
    "product": prodtypes,   "score": pr.cb,   "score_threshold": cbt.thr,   
    "rule precision": cbt.prec, "rule recall": cbt.recall,
  }))

st.write("Logistic Regression:")
st.table(pd.DataFrame({
    "product": prodtypes,   "score": pr.lr,   "score_threshold": lrt.thr,   
    "rule precision": lrt.prec, "rule recall": lrt.recall,
  }))