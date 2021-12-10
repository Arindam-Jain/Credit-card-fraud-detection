import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
from sklearn.metrics import mean_squared_error

df_merge = pd.read_csv('../input/smm-translated-train-and-test/train_translated.csv')

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:14.557944Z","iopub.execute_input":"2021-11-24T04:15:14.558338Z","iopub.status.idle":"2021-11-24T04:15:14.564003Z","shell.execute_reply.started":"2021-11-24T04:15:14.558303Z","shell.execute_reply":"2021-11-24T04:15:14.563197Z"}}
df_merge.columns
df = df_merge.drop(["Unnamed: 0", "Review Date"], axis = 1)
df.isnull().sum()
df = df.sample(frac = 1)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:14.605335Z","iopub.execute_input":"2021-11-24T04:15:14.605643Z","iopub.status.idle":"2021-11-24T04:15:14.62212Z","shell.execute_reply.started":"2021-11-24T04:15:14.605616Z","shell.execute_reply":"2021-11-24T04:15:14.621284Z"}}
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:14.623302Z","iopub.execute_input":"2021-11-24T04:15:14.623534Z","iopub.status.idle":"2021-11-24T04:15:14.630796Z","shell.execute_reply.started":"2021-11-24T04:15:14.623505Z","shell.execute_reply":"2021-11-24T04:15:14.630231Z"}}
df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:14.631779Z","iopub.execute_input":"2021-11-24T04:15:14.632572Z","iopub.status.idle":"2021-11-24T04:15:14.643181Z","shell.execute_reply.started":"2021-11-24T04:15:14.632537Z","shell.execute_reply":"2021-11-24T04:15:14.64246Z"}}
df.columns

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:14.644696Z","iopub.execute_input":"2021-11-24T04:15:14.645517Z","iopub.status.idle":"2021-11-24T04:15:14.663478Z","shell.execute_reply.started":"2021-11-24T04:15:14.645467Z","shell.execute_reply":"2021-11-24T04:15:14.662241Z"}}
df.head()
# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:14.667355Z","iopub.execute_input":"2021-11-24T04:15:14.667582Z","iopub.status.idle":"2021-11-24T04:15:14.674072Z","shell.execute_reply.started":"2021-11-24T04:15:14.667554Z","shell.execute_reply":"2021-11-24T04:15:14.67322Z"}}
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:14.675295Z","iopub.execute_input":"2021-11-24T04:15:14.675905Z","iopub.status.idle":"2021-11-24T04:15:20.606185Z","shell.execute_reply.started":"2021-11-24T04:15:14.67587Z","shell.execute_reply":"2021-11-24T04:15:20.605309Z"}}
df["translate_english"] = df["translate_english"].apply(wordopt)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:20.607557Z","iopub.execute_input":"2021-11-24T04:15:20.607789Z","iopub.status.idle":"2021-11-24T04:15:26.624462Z","shell.execute_reply.started":"2021-11-24T04:15:20.607761Z","shell.execute_reply":"2021-11-24T04:15:26.623807Z"}}
df['final_clean']= df['Country (mentioned)']+ df['Claim']+ df['Source']+ df["translate_english"]
df["final_clean"] = df["final_clean"].apply(wordopt)

x = df["final_clean"]
y = df["Label"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:26.646719Z","iopub.execute_input":"2021-11-24T04:15:26.647052Z","iopub.status.idle":"2021-11-24T04:15:29.817034Z","shell.execute_reply.started":"2021-11-24T04:15:26.647017Z","shell.execute_reply":"2021-11-24T04:15:29.816408Z"}}
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:35.80536Z","iopub.execute_input":"2021-11-24T04:15:35.805939Z","iopub.status.idle":"2021-11-24T04:15:35.817639Z","shell.execute_reply.started":"2021-11-24T04:15:35.805895Z","shell.execute_reply":"2021-11-24T04:15:35.816753Z"}}
pred_lr=LR.predict(xv_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:35.819636Z","iopub.execute_input":"2021-11-24T04:15:35.820215Z","iopub.status.idle":"2021-11-24T04:15:35.834079Z","shell.execute_reply.started":"2021-11-24T04:15:35.820171Z","shell.execute_reply":"2021-11-24T04:15:35.833238Z"}}
LR.score(xv_test, y_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:35.83576Z","iopub.execute_input":"2021-11-24T04:15:35.836306Z","iopub.status.idle":"2021-11-24T04:15:35.84346Z","shell.execute_reply.started":"2021-11-24T04:15:35.836262Z","shell.execute_reply":"2021-11-24T04:15:35.842649Z"}}
mean_squared_error(pred_lr, y_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:35.8453Z","iopub.execute_input":"2021-11-24T04:15:35.845981Z","iopub.status.idle":"2021-11-24T04:15:35.863468Z","shell.execute_reply.started":"2021-11-24T04:15:35.845934Z","shell.execute_reply":"2021-11-24T04:15:35.862686Z"}}
print("Logistic Regression")
print(classification_report(y_test, pred_lr))

# %% [markdown]
# ## Decision Tree Classification

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:35.865123Z","iopub.execute_input":"2021-11-24T04:15:35.865673Z","iopub.status.idle":"2021-11-24T04:15:44.020826Z","shell.execute_reply.started":"2021-11-24T04:15:35.865631Z","shell.execute_reply":"2021-11-24T04:15:44.019809Z"}}
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:44.022038Z","iopub.execute_input":"2021-11-24T04:15:44.022277Z","iopub.status.idle":"2021-11-24T04:15:44.038282Z","shell.execute_reply.started":"2021-11-24T04:15:44.022247Z","shell.execute_reply":"2021-11-24T04:15:44.037327Z"}}
pred_dt = DT.predict(xv_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:44.039654Z","iopub.execute_input":"2021-11-24T04:15:44.040074Z","iopub.status.idle":"2021-11-24T04:15:44.050898Z","shell.execute_reply.started":"2021-11-24T04:15:44.040027Z","shell.execute_reply":"2021-11-24T04:15:44.050164Z"}}
DT.score(xv_test, y_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:44.052092Z","iopub.execute_input":"2021-11-24T04:15:44.0523Z","iopub.status.idle":"2021-11-24T04:15:44.062195Z","shell.execute_reply.started":"2021-11-24T04:15:44.052274Z","shell.execute_reply":"2021-11-24T04:15:44.061317Z"}}
mean_squared_error(pred_dt, y_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:44.063301Z","iopub.execute_input":"2021-11-24T04:15:44.063591Z","iopub.status.idle":"2021-11-24T04:15:44.074461Z","shell.execute_reply.started":"2021-11-24T04:15:44.063547Z","shell.execute_reply":"2021-11-24T04:15:44.073915Z"}}
print('Decision Tree Classification')
print(classification_report(y_test, pred_dt))

# %% [markdown]
# ## Random Forest Classifier

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:44.075452Z","iopub.execute_input":"2021-11-24T04:15:44.07565Z","iopub.status.idle":"2021-11-24T04:15:52.116284Z","shell.execute_reply.started":"2021-11-24T04:15:44.075625Z","shell.execute_reply":"2021-11-24T04:15:52.115721Z"}}
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:52.117547Z","iopub.execute_input":"2021-11-24T04:15:52.117924Z","iopub.status.idle":"2021-11-24T04:15:52.30089Z","shell.execute_reply.started":"2021-11-24T04:15:52.117888Z","shell.execute_reply":"2021-11-24T04:15:52.30009Z"}}
pred_rfc = RFC.predict(xv_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:52.304216Z","iopub.execute_input":"2021-11-24T04:15:52.304434Z","iopub.status.idle":"2021-11-24T04:15:52.489499Z","shell.execute_reply.started":"2021-11-24T04:15:52.304407Z","shell.execute_reply":"2021-11-24T04:15:52.488789Z"}}
RFC.score(xv_test, y_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:52.490536Z","iopub.execute_input":"2021-11-24T04:15:52.490817Z","iopub.status.idle":"2021-11-24T04:15:52.496638Z","shell.execute_reply.started":"2021-11-24T04:15:52.490788Z","shell.execute_reply":"2021-11-24T04:15:52.495926Z"}}
mean_squared_error(pred_rfc, y_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:52.497947Z","iopub.execute_input":"2021-11-24T04:15:52.498147Z","iopub.status.idle":"2021-11-24T04:15:52.512822Z","shell.execute_reply.started":"2021-11-24T04:15:52.498123Z","shell.execute_reply":"2021-11-24T04:15:52.512049Z"}}
print("Random Forest Classifier")
print(classification_report(y_test, pred_rfc))

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:52.514343Z","iopub.execute_input":"2021-11-24T04:15:52.514581Z","iopub.status.idle":"2021-11-24T04:15:53.302813Z","shell.execute_reply.started":"2021-11-24T04:15:52.514553Z","shell.execute_reply":"2021-11-24T04:15:53.301983Z"}}
df_test = pd.read_csv('../input/smm-translated-train-and-test/test_translated.csv')
df_test['final_clean']= df_test['Country (mentioned)']+ df_test['Claim']+ df_test['Source']+ df_test["translate_english"]
df_test["final_clean"] = df_test["final_clean"].apply(wordopt)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:53.303889Z","iopub.execute_input":"2021-11-24T04:15:53.304273Z","iopub.status.idle":"2021-11-24T04:15:53.665878Z","shell.execute_reply.started":"2021-11-24T04:15:53.304239Z","shell.execute_reply":"2021-11-24T04:15:53.665106Z"}}
sub_x_test = df_test["final_clean"]
sub_xv_test = vectorization.transform(sub_x_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:53.666991Z","iopub.execute_input":"2021-11-24T04:15:53.667205Z","iopub.status.idle":"2021-11-24T04:15:53.768688Z","shell.execute_reply.started":"2021-11-24T04:15:53.667179Z","shell.execute_reply":"2021-11-24T04:15:53.767934Z"}}
sub_pred_lr=LR.predict(sub_xv_test)
sub_pred_rfc = RFC.predict(sub_xv_test)
sub_pred_dt = DT.predict(sub_xv_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:53.769884Z","iopub.execute_input":"2021-11-24T04:15:53.7701Z","iopub.status.idle":"2021-11-24T04:15:53.778231Z","shell.execute_reply.started":"2021-11-24T04:15:53.770067Z","shell.execute_reply":"2021-11-24T04:15:53.77751Z"}}
sub_df=pd.read_csv('../input/smmfall21asu/sample_submission.csv')

# %% [code] {"execution":{"iopub.status.busy":"2021-11-24T04:15:53.779298Z","iopub.execute_input":"2021-11-24T04:15:53.779509Z","iopub.status.idle":"2021-11-24T04:15:53.797186Z","shell.execute_reply.started":"2021-11-24T04:15:53.779484Z","shell.execute_reply":"2021-11-24T04:15:53.796316Z"}}
LR_sub = sub_df.copy()
df_LR = pd.DataFrame(sub_pred_lr, columns = ['Predicted'])
LR_sub['Predicted']= df_LR['Predicted']
LR_sub.to_csv('LR_sub.csv',index=False)
DT_sub = sub_df.copy()
df_DT = pd.DataFrame(sub_pred_dt, columns = ['Predicted'])
DT_sub['Predicted']= df_DT['Predicted']
DT_sub.to_csv('DT_sub.csv',index=False)
RFC_sub = sub_df.copy()
df_RFC = pd.DataFrame(sub_pred_rfc, columns = ['Predicted'])
RFC_sub['Predicted']= df_RFC['Predicted']
RFC_sub.to_csv('RFC_sub.csv',index=False)

# %% [code]
