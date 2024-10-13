# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


class RampDetection:

    @classmethod
    def detect_by_rate(cls, data, target="pred_org", ramp="ramp_org", threshold=0.15):
        data[ramp] = False
        for i in range(len(data) - 1):
            window = data[target][i:i + 2]
            min_val = min(window)
            max_val = max(window)
            if max_val > min_val:
                change_rate = (max_val - min_val) / max_val
            else:
                change_rate = (min_val - max_val) / min_val
            is_ramp = abs(change_rate) > threshold
            if is_ramp:
                data.iloc[i:i + 2, data.columns.get_loc(ramp)] = True
        return data
    
    @classmethod
    def get_ramp_events(cls, data, ramp="ramp_org"):
        ramp_events = []
        current_event = []
        for index, row in data.iterrows():
            if row[ramp]:
                current_event.append(row["true"])
            else:
                if current_event:
                    ramp_events.append(current_event)
                    current_event = []
        if current_event:
            ramp_events.append(current_event)
        return ramp_events
    
    @classmethod
    def calculate_event_vol(cls, events):
        spread = [max(event)*10 - min(event)*10 for event in events]
        return sum(spread), np.mean(spread), np.std(spread)
        


if __name__ == "__main__":
    res = pd.read_csv("ramp/res/pred.csv")

    threshold = 0.15

    res = RampDetection.detect_by_rate(res, target="true", ramp="ramp_true", threshold=threshold)
    res = RampDetection.detect_by_rate(res, target="pred_org", ramp="ramp_org", threshold=threshold)
    res = RampDetection.detect_by_rate(res, target="pred_trend", ramp="ramp_trend", threshold=threshold)
    res = RampDetection.detect_by_rate(res, target="pred_seasonal", ramp="ramp_seasonal", threshold=threshold)
    res = RampDetection.detect_by_rate(res, target="pred_residual", ramp="ramp_residual", threshold=threshold)

    res['ramp_stl'] = res['ramp_trend'] & res['ramp_residual']

    tpr_org = sum(res['ramp_true'] & res['ramp_org']) / sum(res['ramp_org'])
    tpr_stl = sum(res['ramp_true'] & res['ramp_stl']) / sum(res['ramp_stl'])

    print(f"TPR ORG: {tpr_org}")
    print(f"TPR STL: {tpr_stl}")

    true_events = RampDetection.get_ramp_events(res, ramp="ramp_true")
    org_events = RampDetection.get_ramp_events(res, ramp="ramp_org")
    stl_events = RampDetection.get_ramp_events(res, ramp="ramp_stl")

    true_vol, true_vol_mean, true_vol_std = RampDetection.calculate_event_vol(true_events)
    org_vol, org_vol_mean, org_vol_std = RampDetection.calculate_event_vol(org_events)
    stl_vol, stl_vol_mean, stl_vol_std = RampDetection.calculate_event_vol(stl_events)

    print(f"TRUE VOL: {true_vol}, TRUE VOL STD: {true_vol_std}")
    print(f"ORG VOL: {org_vol}, ORG VOL STD: {org_vol_std}")
    print(f"STL VOL: {stl_vol}, STL VOL STD: {stl_vol_std}")
    
    # res.to_csv("ramp/res/detection.csv", index=False)