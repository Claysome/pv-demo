# -*- coding: utf-8 -*-

import pandas as pd


def convert_str_to_float(data):
    return [float(pred.strip('[]')) for pred in data]


if __name__ == "__main__":

    data = pd.read_csv("ramp/res/pred.csv")
    data['pred_org'] = convert_str_to_float(data['pred_org'])
    data['pred_trend'] = convert_str_to_float(data['pred_trend'])
    data['pred_seasonal'] = convert_str_to_float(data['pred_seasonal'])
    data['pred_residual'] = convert_str_to_float(data['pred_residual'])

    data.to_csv("ramp/res/pred.csv", index=False)