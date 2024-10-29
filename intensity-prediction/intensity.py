import json
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy.special import softmax

from utils1.time import TimestampAgg
from utils1.ts import TSTransform, CompoundTransform
from utils1.logger import setupLogging
from utils1.dataloader import Dataset
from model.similarity import DTW, Aggregator


class AID:
    def __init__(self):
        loggerName = "AID"
        self._logger = setupLogging('logs', loggerName)
        self._loader = Dataset()

    def _filterCandidate(self, candidateList):
        childSet = set(map(lambda x: x['c'], candidateList))
        parentSet = set(map(lambda x: x['p'], candidateList))
        filteredCand = sorted(filter(lambda x: x['p'] in childSet, candidateList),
                              key=lambda x: x['cnt'],
                              reverse=True)
        return filteredCand

    def _calculateKPIDistance(self,
                              filteredCand,
                              TSDict,
                              kpiList,
                              rowIdx,
                              transformOperations,
                              mpw: int,
                              metricAggFunc=Aggregator.mean_agg,
                              kpiNorm: str = "minmax"):
        def transform(TSDict, cmdbId, kpi, rowIdx):
            srs = pd.Series(TSDict.loc[cmdbId][kpi], index=rowIdx).fillna(0)
            return CompoundTransform(srs, transformOperations)

        for item in filteredCand:
            for kpi in kpiList:
                item[f'dsw-{kpi}'] = DTW.dsw_distance(
                    transform(TSDict, item['c'], kpi, rowIdx),
                    transform(TSDict, item['p'], kpi, rowIdx),
                    mpw=mpw)

        if kpiNorm == "softmax":
            for kpi in kpiList:
                allValues = np.array(
                    list(map(lambda x: x[f'dsw-{kpi}'], filteredCand)))
                x = softmax(allValues)
                assert len(x) == len(filteredCand)
                for idx, candidate in enumerate(filteredCand):
                    candidate[f'normalized-dsw-{kpi}'] = x[idx]
        elif kpiNorm == "minmax":
            for kpi in kpiList:
                allValues = list(map(lambda x: x[f'dsw-{kpi}'], filteredCand))
                maxValue = np.max(allValues)
                minValue = np.min(allValues)
                for candidate in filteredCand:
                    candidate[f'normalized-dsw-{kpi}'] = candidate[f'dsw-{kpi}'] - minValue
                    if maxValue - minValue > 0:
                        candidate[f'normalized-dsw-{kpi}'] /= maxValue - minValue
        else:
            raise NotImplementedError

        # calculate intensity
        for candidate in filteredCand:
            sims_dsw = []
            for kpi in kpiList:
                sims_dsw.append(candidate[f'normalized-dsw-{kpi}'])
            candidate[f'intensity'] = 1-metricAggFunc(sims_dsw)

        filteredCand.sort(key=lambda x: x[f'intensity'], reverse=True)
        return filteredCand

    def eval(self,
             path: str,
             start: str,
             end: str,
             interval: int = 1,
             transformOperations: List[Tuple] = [('ZN',), ("MA", 15)],
             mpw: int = 5):

        # 1. load file
        candidateList, TSDict, cmdbList, kpiList, trace = self._loader.load(
            path,
            tsAggFunc=TimestampAgg.toFreqMinute,
            tsAggFreq=int(interval))

        # 2. preprocess
        # filter candidate
        candidateList = self._filterCandidate(candidateList)

        # filter data point
        def genDate(datestr):
            return f"{datestr[:4]}-{datestr[4:6]}-{datestr[6:8]}"

        rowIdx = pd.date_range(f"{genDate(start)} 00:00:00",
                               f"{genDate(end)} 23:59:00", freq=f'{interval}T')

        # 3. Calculate intensity
        intensityList = self._calculateKPIDistance(candidateList, TSDict, kpiList, rowIdx,
                                                   transformOperations=transformOperations,
                                                   mpw=mpw,
                                                   metricAggFunc=Aggregator.mean_agg)

        # remove unnecessary attributes
        intensityList = list(map(
            lambda x: {"c": x["c"],
                       "p": x["p"],
                       "intensity": x["intensity"]},
            intensityList)
        )

        separator = "_"
        new_dict = {}
        for item in intensityList:
            child = item.get('c')
            parent = item.get('p')
            key = parent + separator + child
            new_dict[key] = item.get('intensity')

        def apply_intensity(row):
            p_id = row['parent_id']
            c_id = row['child_id']
            key_id = p_id + separator + c_id
            return new_dict.get(key_id, 0)

        trace['intensity'] = trace.apply(lambda row: apply_intensity(row), axis=1)

        df3 = pd.DataFrame(trace['s_t'].to_list(), columns=['source', 'target'])
        trace = pd.concat([trace, df3], axis=1, ignore_index=False, sort=False)
        trace = trace.dropna()
        trace.to_csv("D:/sdprca-main/data/intensity/op/trace_with_intensity.csv", encoding='utf-8', index=False)
        trace.to_pickle("D:/sdprca-main/data/intensity/op/trace_with_intensity.pkl")

        return intensityList


if __name__ == "__main__":
    aid = AID()
    intensity = aid.eval("D:/sdprca-main/data/intensity/ip/1.pkl",
                         start="20191006",
                         end='20191008')
        # "../data/intensity/ip/admin-order_abort_1011.pkl",
        #                  start="20191010",
        #                  end='20191012')
    with open("D:/sdprca-main/data/intensity/op/intensity.json", 'w') as f:
        json.dump(intensity, f, indent=4)
