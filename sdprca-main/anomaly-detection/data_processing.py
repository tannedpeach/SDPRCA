import pickle
import pandas as pd

file = r'D:/sdprca-main/data/detection/ip/1.pkl'
with open(str(file),'rb') as f:
    pkl_data = pickle.load(f)
df = pd.DataFrame(pkl_data)
df = df.explode(['timestamp', 'latency', 'http_status', 'cpu_use', 'net_send_rate', 'net_receive_rate', 'mem_use_percent', 'mem_use_amount', 'file_read_rate', 'file_write_rate', 'endtime', 's_t']).reset_index(drop=True)

df3 = pd.DataFrame(df['s_t'].to_list(), columns=['source', 'target'])
df = pd.concat([df, df3], axis=1, ignore_index=False, sort=False)

def save_dict(data, name):
    with open(name , 'wb') as f:
        pickle.dump(data, f)
print(df)

save_dict(df,r'D:/sdprca-main/data/detection/op/pkl_3_data.pkl')