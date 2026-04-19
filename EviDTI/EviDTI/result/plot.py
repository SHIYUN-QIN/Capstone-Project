import argparse
import os.path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import scipy
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr

from itertools import cycle
import matplotlib.lines as mlines
from matplotlib.legend import Legend

def draw_pearson_scatter(y_true, y_pred, label):
    fig, ax = plt.subplots(figsize=(5, 5))

    xy = np.vstack([y_true, y_pred])
    z = gaussian_kde(xy)(xy)

    ax.scatter(y_true, y_pred, c=z)

    ax.spines['top'].set_linewidth(1.3)
    ax.spines['bottom'].set_linewidth(1.3)
    ax.spines['right'].set_linewidth(1.3)
    ax.spines['left'].set_linewidth(1.3)
    ax.tick_params(direction='in', which='both', labelsize=10)

    ax.set_xlabel(f'experimental {label}', fontdict={'family': 'arial', 'size': 14})
    ax.set_ylabel(f'predict {label}', fontdict={'family': 'arial', 'size': 14})

    ax.set_xlim([-5.2, 6.2])
    ax.set_ylim([-5.2, 6.2])

    r, p = pearsonr(y_true, y_pred)

    ax.text(-4, 5.5, f"r = {round(r, 3)}", size=8)
    ax.text(-4, 5, f"P = {round(p, 3)}", size=8)
    ax.text(-4, 4.5, f"N = {len(y_true)}", size=8)

    plt.show()
    return fig


def load_file(file):
    _, ext = os.path.splitext(file)
    sep = "," if ext == ".csv" else "\t"
    return pd.read_csv(file, sep=sep)


def main(input_file, pred_file, label, save_file):
    assert label in ["Kcat", "Km"]

    _pred = load_file(pred_file)
    _pred.columns = ["Kcat", "Km"]

    _input = load_file(input_file)

    y_true = _input[label].values
    y_pred = _pred[label].values

    fig = draw_pearson_scatter(y_true, y_pred, label)
    fig.savefig(save_file, bbox_inches='tight', pad_inches=.3, dpi=1200)

def draw_pair_bar(data: ArrayLike, colors, group_name, item_name, step=1.5, width=0.2):
    fig, ax = plt.subplots(figsize=(16, 9))

    n = data.shape[1]
    x = []
    xticks = []

    for center, ds in enumerate(data):
        if n % 2 == 0:

            c = center + 1
            m = (n / 2) - 0.5
            _x = np.arange(c - m * width, c + (n / 2) * width, width) + center * step
            x.append(_x)
            xticks.append((center + center * step) + 1)

    x = np.hstack(x)
    ax.bar(x, data.flatten(), width=width, color=np.tile(colors, n), edgecolor='white', alpha=0.8)

    ax.set_ylim([0, 1])
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.tick_params(direction='in', which='both', labelsize=14, width=0.3)

    ax.set_xticks(xticks)
    ax.set_xticklabels(group_name)

    legend = [mlines.Line2D([0], [0], color=c, lw=12) for c in colors]
    leg = Legend(ax, legend, labels=item_name, ncol=1,
                 fontsize=14,
                 edgecolor='white', borderpad=0.6)
    ax.add_artist(leg)
    plt.show()
    return fig

def draw_box_plot(data: ArrayLike):
    fig, ax = plt.subplots(figsize=(5, 6))

    # rectangular box plot
    bplot = ax.violinplot(
        data,
        vert=True,  # vertical box alignment
        showmeans=False,
        showmedians=True
    )

    colors = ['#bec936', '#4e7ca1', '#ed556a','#ed556a']
    for patch, color in zip(bplot['bodies'], colors):
        patch.set_facecolor(color)

    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.tick_params(direction='in', which='both', labelsize=14, width=0.3)

    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(data))])
    ax.set_xticklabels(['Random', 'Alternative', 'Prefer', 'P'])

    ax.yaxis.grid(True)
    plt.show()
    return fig

import seaborn as sns
# from statannotations.Annotator import Annotator

def sns_draw_box_plot(data):
    df = data
    sns.set_theme(style="whitegrid", palette="pastel")

    # Load the example tips dataset
    # data = sns.load_dataset("tips")

    # Draw a nested boxplot to show bills by day and time
    # x = "sample"
    # y = "u"
    order = ['TP', 'FN', 'FP', 'TN']
    fig, ax = plt.subplots(figsize=(5, 4), dpi=100, facecolor="w")
    ax = sns.boxplot(data=df,order=order,ax=ax,
                     fliersize=3,
                flierprops = {'marker': 'o',  # 异常值形状
                  'markerfacecolor': 'gray',  # 形状填充色
                  'color': 'black',  # 形状外廓颜色
                  },
    )

    pairs = [("TP", "FN"), ("TP", "FP"), ("TN", "FN"), ("TN", "FP")]

    annotator = Annotator(ax, pairs,data=df,order=order)
    annotator.configure(test='Mann-Whitney', text_format='star', line_height=0.03, line_width=1)
    annotator.apply_and_annotate()

    # sns.despine(offset=10, trim=True)
    # plt.show()
    # return fig.get_figure()
    ax.tick_params(which='major', direction='in', length=3, width=1., labelsize=14, bottom=False)
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_linewidth(2)
    ax.grid(axis='y', ls='--', c='gray')
    ax.set_axisbelow(True)
    plt.show()
    return fig.get_figure()

def sns_draw_line_plot(data):

    sns.set_theme(font='arial',font_scale=2)
    sns.set_style('ticks')#ticks whitegrid
    plt.figure(figsize=(10, 8))
    # dots = sns.load_dataset("dots")
    # Define the palette as a list to specify exact values
    # palette = sns.color_palette("rocket_r")
    # Plot the lines on two facets
    sns.set_context(rc={'lines.linewidth': 2.5})
    fig = sns.lineplot(data=data,x='index', y='davis', label='davis')
    sns.lineplot(data=data, x='index', y='kiba', label='kiba')
    sns.lineplot(data=data, x='index', y='drugbank', label='drugbank')
    # sns.despine(offset=10)
    # fig.set_xticks(ticks = [0,10,20,30,40,50,60,70,80,90,100])
    # fig.set_xticks(ticks=["0", "2", "4", "6", "8", "10", "12", "14", "16", "18", "20"])
    # fig.set_xticks(ticks=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19])
    fig.set_xticks(ticks=[1, 2, 4, 6, 8, 10, 12, 14, 16, 18,20])
    plt.legend(loc="lower left")
    plt.ylabel('Acc')
    plt.xlabel('Interval')
    plt.xlim(1, 21)
    plt.ylim(0.5, 1.03)

    sns.despine(top=True, right=True, left=False, bottom=False)
    # fig.legend(loc='center right')
    plt.show()
    return fig.get_figure()

def sns_draw_relplot(data):
    sns.set_theme(font='arial',font_scale=2) # style="ticks"
    sns.set_style('ticks')# "darkgrid"
    # plt.figure(figsize=(10, 20))
    sns.set_context(rc={'lines.linewidth': 2.5})
    sns.despine(top=True, right=True, left=True, bottom=True)
    # sns.set_theme(style="white", palette="deep")
    plot = sns.relplot(x = 'Rank',y='Hit rate',hue='event', kind="line",
                      legend=False, data=data,palette=['red', 'blue'],
                       facet_kws={'legend_out': True},
                       line_kws={'alpha': 0.2})
    # 设置坐标轴距离
    # sns.despine(offset=10)
    plt.xlim(0,20)
    # plt.xticks = 0,2,4,6,8,10,12,14,16,18
    plt.ylim(0.3,1.02)
    plot.fig.set_size_inches(10, 8)
    plot.ax.set_xticks(ticks = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    # plot.set_xticklabels([0,3,6,9,12,15,18,21,24])
    plt.legend(labels=["uncertainty", "probability"],loc="upper right")
    # plt.grid(True)
    plt.show()
    return plot.fig


# import numpy as np
# import matplotlib.pyplot as plt


def plot_ofr_with_variance(ofr_values):
    """
    Plots the average OFR values with variance shaded area.

    Parameters:
    ofr_values (np.ndarray): A 2D numpy array of shape (n_experiments, n_thresholds)
                             containing OFR values for different experiments and thresholds.
    """

    evidential_value = ofr_values[['1','2','3','4']].dropna()#'1','2','3','4'
    no_evidential_value = ofr_values[['6','7']].dropna() #'5','6','7','8'
    # Calculate the mean and variance across experiments for each threshold
    evidential_mean_ofr = np.mean(evidential_value, axis=1)
    evidential_variance_ofr = np.std(evidential_value, axis=1)
    no_evidential_mean_ofr = np.mean(no_evidential_value, axis=1)
    no_evidential_variance_ofr = np.std(no_evidential_value, axis=1)

    # Define the thresholds (assuming 10 thresholds)
    thresholds = np.arange(0.1, 0.0, -0.01)
    evidential_data = pd.DataFrame({
        'Threshold': thresholds,
        'Mean OFR': evidential_mean_ofr,
        'Variance': evidential_variance_ofr
    })
    no_evidential_data = pd.DataFrame({
        'Threshold': thresholds,
        'Mean OFR': no_evidential_mean_ofr,
        'Variance': no_evidential_variance_ofr
    })
    sns.set_theme(font='arial', font_scale=2)
    sns.set_style('ticks')

    # Plot the average OFR values
    plt.figure(figsize=(10, 8))
    # plt.plot(thresholds, evidential_mean_ofr, label='Evidential Average OFR', color='b', marker='o')
    # plt.plot(thresholds, no_evidential_mean_ofr, label='ProbabilityAverage OFR', color='r', marker='o')
    # Line plot for mean OFR
    plot = sns.lineplot(x='Threshold', y='Mean OFR', data=evidential_data, color='r',marker='o', label='Uncertainty Average OFR')
    sns.lineplot(x='Threshold', y='Mean OFR', data=no_evidential_data, marker='o', label='Probability Average OFR')

    # Plot the variance as shaded area
    plt.fill_between(thresholds, evidential_mean_ofr - evidential_variance_ofr,
                     evidential_mean_ofr + evidential_variance_ofr, color='r', alpha=0.2,
                     label='Uncertainty std')
    plt.fill_between(thresholds, no_evidential_mean_ofr - no_evidential_variance_ofr,
                     no_evidential_mean_ofr + no_evidential_variance_ofr, color='b', alpha=0.2,
                     label='Probability std')
    plt.gca().invert_xaxis()
    plt.xticks(thresholds)
    # 设置坐标轴距离
    # sns.despine(offset=10)
    plt.legend(loc="lower left")
    # Add labels and title
    plt.xlabel('Threshold')
    plt.ylabel('OFR')
    plt.title('DrugBank')

    # Show plot
    plt.grid(True)
    plt.show()
    return plot.get_figure()


def plot_tables_with_mean_and_std(tables, labels):
    """
    绘制多个数据表的折线图，显示平均值和标准差。

    参数:
    tables (list of pd.DataFrame): 包含多个数据表的列表。
    labels (list of str): 每个数据表的标签列表。
    """
    plot = plt.figure(figsize=(14, 8))
    sns.set_theme(font='arial', font_scale=2)
    sns.set_style('ticks')# whitegrid ticks

    markers = ['o', 's', '^', 'D', 'v']  # 不同数据表的标记
    linestyles = ['-', '--', '-.', ':', '-']  # 不同数据表的线型

    for i, (df, label) in enumerate(zip(tables, labels)):
        mean = df.mean(axis=1)
        std = df.std(axis=1)

        # 绘制平均值
        plt.plot(range(1, len(mean) + 1), mean.values, marker=markers[i % len(markers)],
                 linestyle=linestyles[i % len(linestyles)], label=f'{label}')

        # 绘制标准差阴影
        plt.fill_between(range(1, len(mean) + 1), mean - std, mean + std, alpha=0.2)

    # 添加图例和标题
    # plt.legend()
    # plt.title('Mean and Standard Deviation for Multiple Tables')
    # sns.despine(offset=10)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.xlabel('Confidence intervals(%)')
    plt.ylabel('Accuracy')
    labels = [' ','0-5', '5-10', '10-15', '15-20',
              '20-25','25-30','30-35','35-40',
              '40-45','45-50','50-55','55-60',
              '60-65','65-70','70-75','75-80',
              '80-85','85-90','90-95','95-100']
    # plt.xlim(-0.5, len(labels) - 0.5)
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45, ha='right')

    # 显示图形
    plt.show()
    return plot.get_figure()

def plot_bar_with_variance(data):
    """
    绘制柱状图并显示每个值的方差。

    参数:
    data (dict): 包含多个组数据的字典，每组数据有相同数量的值。
    group_labels (list): 每组数据的标签列表。
    """
    # 转换为DataFrame
    # df = pd.DataFrame(data, columns=group_labels)
    data_edti_mean = data['edti'].iloc[0,1:].values
    data_edti_std = data['edti'].iloc[1,1:].values
    data_edti = {
        'Metric': ['Accuracy', 'Recall', 'Precision', 'MCC', 'F1 Score', 'AUC', 'AUPR'],
        'Mean': data_edti_mean,
        'Std': data_edti_std
    }

    data_int_mean = data['int'].iloc[0,1:].values
    data_int_std = data['int'].iloc[1, 1:].values
    data_int = {
        'Metric': ['Accuracy', 'Recall', 'Precision', 'MCC', 'F1 Score', 'AUC', 'AUPR'],
        'Mean': data_int_mean,
        'Std': data_int_std
    }
    data_gcn_mean = data['gcn'].iloc[0, 1:].values
    data_gcn_std = data['gcn'].iloc[1, 1:].values
    data_gcn = {
        'Metric': ['Accuracy', 'Recall', 'Precision', 'MCC', 'F1 Score', 'AUC', 'AUPR'],
        'Mean': data_gcn_mean,
        'Std': data_gcn_std
    }

    # data_no2d_mean = data['no2d'].iloc[0, 1:].values
    #     # data_no2d_std = data['no2d'].iloc[1, 1:].values
    #     # data_no2d = {
    #     #     'Metric': ['Accuracy', 'Recall', 'Precision', 'MCC', 'F1 Score', 'AUC', 'AUPR'],
    #     #     'Mean': data_no2d_mean,
    #     #     'Std': data_no2d_std
    #     # }
    #     # data_no3d_mean = data['no3d'].iloc[0, 1:].values
    #     # data_no3d_std = data['no3d'].iloc[1, 1:].values
    #     # data_no3d = {
    #     #     'Metric': ['Accuracy', 'Recall', 'Precision', 'MCC', 'F1 Score', 'AUC', 'AUPR'],
    #     #     'Mean': data_no3d_mean,
    #     #     'Std': data_no3d_std
    #     # }

    df1 = pd.DataFrame(data_edti)
    df1['Table'] = 'EviDTI'

    # df2 = pd.DataFrame(data_no2d)
    df2 = pd.DataFrame(data_int)
    # df2['Table'] = 'EviDTI w/o drug 2D'
    df2['Table'] = 'Protein integer'
    #
    # df3 = pd.DataFrame(data_no3d)
    # df3['Table'] = 'EviDTI w/o drug 3D'
    df3 = pd.DataFrame(data_gcn)
    df3['Table'] = 'Drug 2D GCN'
    df_combined = pd.concat([df2, df3, df1])

    # 自定义调色板
    palette = {
        'EviDTI': '#ea5455',  # 设置df1为红色
        # 'EviDTI w/o drug 2D': '#ffd460',  # 使用默认调色板的颜色
        # 'EviDTI w/o drug 3D': '#f07b3f'  # 使用默认调色板的颜色
        'Protein integer': '#c4edde',
        'Drug 2D GCN': '#7ac7c4'
    }

    # 绘制柱状图
    sns.set_theme(font='arial', font_scale=2)#, palette="pastel"
    sns.set_style('ticks')  # whitegrid ticks
    plot = plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Metric', y='Mean', hue='Table', data=df_combined, ci=None, palette=palette)#, palette=palette

    # 添加误差条
    n_tables = df_combined['Table'].nunique()
    n_metrics = df_combined['Metric'].nunique()

    for i, table in enumerate(df_combined['Table'].unique()):
        table_data = df_combined[df_combined['Table'] == table]
        x = np.arange(len(table_data['Metric']))
        x = x + (i - (n_tables - 1) / 2) * 0.26  # 调整误差条位置，使其对应到各个柱上
        ax.errorbar(x=x, y=table_data['Mean'], yerr=table_data['Std'], fmt='none', c='black', capsize=5)

    # 添加标题和标签
    plt.title('Davis')
    sns.despine(offset=10)
    plt.legend(loc='upper center', prop={'size': 20}) #bbox_to_anchor=(1.05, 1),
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    ax.tick_params(axis='x', labelsize=26)
    ax.tick_params(axis='y', labelsize=26)
    plt.ylim(50, 100)

    # 显示图形
    plt.show()
    return plot.get_figure()


from scipy.stats import dirichlet,beta
def draw_Dirichlet():
    # 参数alpha定义二元狄利克雷分布的形状参数（实际上是Beta分布）
    alpha = [2, 5]

    # 生成数据
    n_points = 10000
    data = np.random.beta(alpha[0], alpha[1], n_points)

    # 定义网格
    x = np.linspace(0, 1, 100)

    # 计算密度
    density = beta.pdf(x, alpha[0], alpha[1])

    # 绘制密度图
    plt.figure(figsize=(12, 6))
    plt.plot(x, density, label=f'α1={alpha[0]}, α2={alpha[1]}',color = 'slategray')
    # sns.kdeplot(x,density)
    plt.legend(loc='upper right',fontsize=28)
    plt.fill_between(x, density, alpha=0.5,color = 'slategray')
    # .axis['right'].set_visible["false"]
    # plt.title('Beta Distribution Density Plot')
    plt.xlabel('probability', fontsize=28)
    plt.ylabel('Density', fontsize=28)
    plt.xticks(fontsize=28)
    plt.yticks(fontsize=28)
    # plt.grid(True)
    # plt.legend()
    plt.show()

def plot_hit_ratio_chart(y):
    """
    使用Seaborn绘制折线图

    参数:
    x (list or np.array): X轴数据
    y (list or np.array): Y轴数据
    """
    # 创建数据框
    x = np.arange(0,101)
    data = {'Top N%': x, 'Binding sites hit ratio': y.values.squeeze()}

    # 使用seaborn绘制折线图
    plot = sns.lineplot(x='Top N%', y='Binding sites hit ratio', data=data)
    # plt.legend(loc='upper left', fontsize=34)
    plt.grid(True)

    # 显示图表
    plt.show()
    return plot.get_figure()


if __name__ == "__main__":

    data1 = pd.read_csv(r".\evidental\box\davis_result_22-09_10-08-56.csv")
    # data2 = pd.read_csv(r".\evidental\box\drugbank21-08_08-29-28result.csv")
    # data3 = pd.read_csv(r".\evidental\box\kiba_result_22-09_10-11-52.csv")
    # data4 = pd.read_excel(r"D.\evidental\散点图.xlsx")
    # data5 = pd.read_csv(r".\DTI_drugbank\success_rate.txt")
    # data6 = pd.read_csv(r".\case study\independent_v1.csv")
    # data7 = pd.read_csv(r".\evidental\OFR_drugbank.csv")#.dropna()
    # stat, p_value = scipy.stats.ttest_ind(data1['TN'],
    #                                       data1['FP'],
    #                                       equal_var=False)
    # print(p_value)
    # f = sns_draw_box_plot(data5)
    # f = sns_draw_line_plot(data4)
    # f = sns_draw_relplot(data6)
    # f.savefig(r".\evidental\fig\hitrate_V4.png", bbox_inches='tight', pad_inches=.3, dpi=1200)

    # Example usage:
    # Assume we have OFR values for 5 experiments and 10 thresholds
    # f = plot_ofr_with_variance(data7)
    # f.savefig(r".\evidental\fig\ofr_drugbank.png", bbox_inches='tight', pad_inches=.3, dpi=1200)

    # data_8_1 = pd.read_excel(r".\evidental\drugbank_intervals.xlsx")
    # select_cols = data_8_1.columns[1:4]
    # data_8_1 = data_8_1[select_cols]
    # data_8_2 = pd.read_excel(r".\evidental\kiba_intervals.xlsx")
    # select_cols = data_8_2.columns[1:4]
    # data_8_2 = data_8_2[select_cols]
    # data_8_3 = pd.read_excel(r".\evidental\davis_intervals.xlsx")
    # select_cols = data_8_3.columns[1:5]
    # data_8_3 = data_8_3[select_cols]
    # f = plot_tables_with_mean_and_std([data_8_1, data_8_2, data_8_3], ['DrugBank', 'KIBA', 'Davis'])
    # f.savefig(r".\evidental\fig\intervals.png", bbox_inches='tight', pad_inches=.3, dpi=600)

    # data_9_1 = pd.read_excel(r".\evidental\drugbank_ablation_result.xlsx",sheet_name=['edti','int','gcn'])#'edti','no2d','no3d'
    # data_9_2 = pd.read_excel(r".\evidental\KIBA_ablation_result.xlsx",sheet_name=['edti','int','gcn'])
    # data_9_3 = pd.read_excel(r".\evidental\davis_ablation_result.xlsx",sheet_name=['edti','int','gcn'])#'edti','int','gcn'
    # f = plot_bar_with_variance(data_9_3)
    # f.savefig(r".\fig\ablation2_Davis.eps", bbox_inches='tight', pad_inches=.3, dpi=600)

    # f= draw_Dirichlet()

    # data_10 = pd.read_csv(r".\att_case\success_rate.txt",header=None)
    # f = plot_hit_ratio_chart(data_10)
    # f.savefig(r".\fig\success_rate.png", bbox_inches='tight', pad_inches=.3, dpi=600)
