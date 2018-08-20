
# coding: utf-8

# In[19]:


import pandas as pd
df_orig = pd.read_csv('train.csv')
df_orig.head()


# In[23]:


sex_dum = pd.get_dummies(df_orig['Sex'],drop_first=True)
df_using = pd.concat((df_orig, sex_dum), axis = 1)
df_using = df_using.drop('Sex',axis = 1)


# In[24]:


emb_dum = pd.get_dummies(df_orig['Embarked'],drop_first=True)
df_using = pd.concat((df_using, emb_dum), axis = 1)
df_using = df_using.drop('Embarked',axis = 1)

df_using.head()


# In[25]:


df_using.isnull().sum()


# In[32]:


df_tmp = df_using.dropna()
df_tmp = df_tmp.drop('PassengerId',axis = 1)
df_tmp = df_tmp.drop('Name',axis = 1)
df_tmp = df_tmp.drop('Ticket',axis = 1)
df_tmp = df_tmp.drop('Cabin',axis = 1)
df_tmp.corr()


# In[34]:


import matplotlib.pyplot as plt
plt.style.use('ggplot')

df_draw = df_tmp[df_tmp.Survived==1]
df_age = df_draw.iloc[:,2]
df_sex = df_draw.iloc[:,6]
plt.scatter(df_age,df_sex, color='#cc6699',alpha=0.5)

df_draw = df_tmp[df_tmp.Survived==0]
df_age = df_draw.iloc[:,2]
df_sex = df_draw.iloc[:,6]
plt.scatter(df_age,df_sex, color='#6699cc',alpha=0.5)

plt.show()



# In[36]:


def title_flag(name_df):
    ans_df = pd.DataFrame(columns={'miss','mrs','master','mr'})
    
    for name in name_df:
        if 'Miss' in name:
            tmp_df = pd.DataFrame([1,0,0,0],columns={'miss','mrs','master','mr'})
        elif 'Mrs' in name:
            tmp_df = pd.DataFrame([0,1,0,0],columns={'miss','mrs','master','mr'})
        elif 'Master' in name:
            tmp_df = pd.DataFrame([0,0,1,0],columns={'miss','mrs','master','mr'})
        else:
            tmp_df = pd.DataFrame([0,0,0,1],columns={'miss','mrs','master','mr'})
        
        ans_df = ans_df.append(tmp_df, ignore_index=True)
    return ans_df


