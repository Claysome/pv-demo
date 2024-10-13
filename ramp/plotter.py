# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import pandas as pd


def plot_ramp_org(data, title="Ramp Detection"):
    # 创建图表和子图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(data.index, data['true'], label='True Value', color='blue')
    ax.plot(data.index, data['pred_org'], label='Predicted Value', color='gray')

    for i in range(len(data)):
        if data['ramp_true'].iloc[i]:
            ax.axvspan(data.index[i], data.index[i+1], color='yellow', alpha=0.3)

    for i in range(len(data)):
        if data['ramp_org'].iloc[i]:
            ax.axvspan(data.index[i], data.index[i+1], color='green', alpha=0.3)
    
    # 添加图例和标题
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Power')

    # 显示图表
    plt.xticks(rotation=45)  # x轴标签旋转
    plt.tight_layout()
    plt.show()


def plot_ramp_stl(data, title="Ramp Detection"):
    # 创建图表和子图
    fig, ax = plt.subplots(figsize=(10, 6))
    data['pred_stl'] = data['pred_trend'] + data['pred_residual'] + data['pred_seasonal']
    
    ax.plot(data.index, data['true'], label='True Value', color='blue')
    ax.plot(data.index, data['pred_stl'], label='Predicted Value', color='gray')

    for i in range(len(data)):
        if data['ramp_true'].iloc[i]:
            ax.axvspan(data.index[i], data.index[i+1], color='yellow', alpha=0.3)

    for i in range(len(data)):
        if data['ramp_stl'].iloc[i]:
            ax.axvspan(data.index[i], data.index[i+1], color='blue', alpha=0.3)
    
    # 添加图例和标题
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Power')

    # 显示图表
    plt.xticks(rotation=45)  # x轴标签旋转
    plt.tight_layout()
    plt.show()
    
    


if __name__ == "__main__":

    # time_range = pd.date_range(start='2019-10-24 00:00:00', end='2019-12-31 23:45:00', freq='15min')
    # detection = pd.read_csv("ramp/res/detection.csv")

    # detection['DATETIME'] = pd.to_datetime(detection['DATETIME'])  # 确保时间格式正确
    # detection.set_index('DATETIME', inplace=True)

    # result_df = pd.DataFrame(index=time_range, columns=detection.columns).fillna(0)
    # result_df.update(detection)
    
    # columns_to_replace = ['ramp_true', 'ramp_org', 'ramp_trend', 'ramp_seasonal', 'ramp_residual', 'ramp_stl']
    # result_df[columns_to_replace] = result_df[columns_to_replace].replace(0, False)
    # result_df.to_csv("ramp/res/full.csv", index=True)

    data = pd.read_csv("ramp/res/full.csv")
    data['DATETIME'] = pd.to_datetime(data['DATETIME'])  # 确保时间格式正确
    data.set_index('DATETIME', inplace=True)
    print(data)
    plot_ramp_org(data, title='Ramp Detection with LSTM Model')
    plot_ramp_stl(data, title='Ramp Detection with STL-LSTM Model')
    