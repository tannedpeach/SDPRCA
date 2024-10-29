import numpy as np
import pandas as pd
from datetime import datetime

class Dataset:

    def loadRawData(self, filename):
        trace = pd.read_pickle(filename)
        trace = pd.DataFrame(trace)
        trace = trace.sample(n=60)
        trace = trace.explode(['timestamp', 'latency', 'http_status', 'endtime', 's_t']).reset_index(drop=True)
        df3 = pd.DataFrame(trace['s_t'].to_list(), columns=['source', 'target'])
        trace = pd.concat([trace, df3], axis=1, ignore_index=False, sort=False)
        trace['call_num_sum'] = trace.groupby(['source', 'target']).cumcount() + 1
        trace.rename(columns={'source': 'parent_id', 'target': 'child_id', 'timestamp': 'ts'}, inplace=True)
        trace = trace[trace['parent_id'] != trace['child_id']]
        trace['latency_max'] = trace.groupby(['parent_id', 'child_id'])['latency'].transform('max')
        trace['ts'] = trace['ts'].astype(float).div(1000).round(2).div(1000).round(2)
        print(datetime.fromtimestamp(trace['ts'].min()))
        print(datetime.fromtimestamp(trace['ts'].max()))
        # trace['ts'] = trace['ts'].astype(float).div(1000).round(2)  # for route_delay_0421
        # trace['ts'] = trace['ts'].div(1000).round(2)
        print(trace.dtypes)
        print(trace.head())
        return trace

    def getCandidateListByDF(self, df):
        df = df.groupby(['parent_id', 'child_id']).agg(
            {'call_num_sum': np.sum}).reset_index()
        candidateList = []
        for i in range(df.shape[0]):
            candidateList.append({
                'c': df.iloc[i]['child_id'],
                'p': df.iloc[i]['parent_id'],
                'cnt': df.iloc[i]['call_num_sum']
            })
        return candidateList

    def getTSDictByDF(self, df, tsAggFunc, tsAggFreq):
        # need to process trace
        cmdbList = list(df['child_id'].unique())

        # for tracerca
        df['latency_sum'] = df['latency'] * df['call_num_sum']
        df['ts'] = df['ts'].apply(tsAggFunc, args=(
            tsAggFreq,)).apply(pd.to_datetime)
        tmpdf = df.groupby(['child_id', 'ts']).agg({
            'call_num_sum': np.sum,
            'latency_sum': np.sum,
            'latency_max': np.max,
        })
        tmpdf['latency_avg'] = tmpdf['latency_sum'] / \
                               tmpdf['call_num_sum']
        tmpdf.drop(columns=['latency_sum'], inplace=True)

        TSDict = tmpdf
        return TSDict, cmdbList

    def load(self, fileName, tsAggFunc, tsAggFreq):
        trace = self.loadRawData(fileName)
        candidateList = self.getCandidateListByDF(trace)
        TSDict, cmdbList = self.getTSDictByDF(trace, tsAggFunc, tsAggFreq)
        kpiList = list(TSDict.columns)
        return candidateList, TSDict, cmdbList, kpiList, trace
