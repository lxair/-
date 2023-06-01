import sys
from pathlib import Path
import qlib
import pandas as pd
from qlib.config import REG_CN
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha360
from qlib.utils import init_instance_by_config
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.contrib.report import analysis_model, analysis_position
# from qlib.contrib.evaluate import (
#     backtest as normal_backtest,
#     risk_analysis,
# )
from qlib.utils import exists_qlib_data, init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import flatten_dict
from qlib.data.dataset.loader import QlibDataLoader
from qlib.contrib.data.handler import Alpha158   #Alpha158内置指标体系
from qlib.data.dataset.loader import QlibDataLoader
import qlib
from qlib.contrib.data.handler import Alpha158   #Alpha158内置指标体系


provider_uri = "./qlib_data/cn_data"  # 原始行情数据存放目录
qlib.init(provider_uri=provider_uri, region=REG_CN)  # 初始化
market = "csi100"
benchmark = "SH000300"

 #数据处理器参数配置
data_handler_config = {
    # "start_time": "2008-01-01",
    # "end_time": "2020-08-01",
    # "fit_start_time": "2008-01-01",
    # "fit_end_time": "2014-12-31",
    "start_time": "2020-01-01",
    "end_time": "2020-02-21",
    "fit_start_time": "2020-01-01",  # 模型跑数据的开始时间
    "fit_end_time": "2020-01-31",
    "instruments": market,
    'infer_processors': [
                                    {'class': 'RobustZScoreNorm',
                                     'kwargs': {'fields_group': 'feature', 'clip_outlier': True}},
                                    {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}],
    
    'learn_processors': [{'class': 'DropnaLabel'},
                                                     
                                                     # 对预测的目标进行截面排序处理
                                    {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}],
                                
                                # 预测的目标
                                'label': ['Ref($close, -1) / $close - 1']  # 下一日的收益率.
}
    

# 任务参数配置
task = {
    "model": {  # 模型参数配置
        # 模型类
        "class": "TransGANModel",
        # 模型类所在模块
        "module_path": "qlib.contrib.model.transgandemo",
        "kwargs": {  # 模型超参数配置
            
            
        }, 
    },
    "dataset": {  # 　因子库数据集参数配置
        # 数据集类，是Dataset with Data(H)andler的缩写，即带数据处理器的数据集
        "class": "TSDatasetH",
        # 数据集类所在模块
        "module_path": "qlib.data.dataset",
        "kwargs": {  # 数据集参数配置
            "handler": {  # 数据集使用的数据处理器配置
                #"class": "Alpha158",  # 数据处理器类，继承自DataHandlerLP
                "module_path": "qlib.contrib.data.handler",  # 数据处理器类所在模块
                "class": "Alpha158",
                "kwargs": data_handler_config,  # 数据处理器参数配置
            },
            "segments": {  # 数据集划分标准
                # "train": ("2008-01-01", "2014-12-31"), # 此时段的数据为训练集
                # "valid": ("2015-01-01", "2016-12-31"), # 此时段的数据为验证集
                # "test": ("2017-01-01", "2020-08-01"),  # 此时段的数据为测试集
                "train": ("2020-01-01", "2020-01-31"),  # 此时段的数据为训练集
                "valid": ("2020-01-31", "2020-02-20"),  # 此时段的数据为验证集
                "test": ("2020-02-20", "2020-02-21"),  # 此时段的数据为测试集
            },
        },
    },

}

# 实例化模型对象
model = init_instance_by_config(task["model"])

# 实例化因子库数据集，从基础行情数据计算出的包含所有特征（因子）和标签值的数据集。
dataset = init_instance_by_config(task["dataset"])  # DatasetH


# start exp to train model
with R.start(experiment_name="train_model"):
    R.log_params(**flatten_dict(task))
    model.fit(dataset)
    R.save_objects(trained_model=model)
    rid = R.get_recorder().id

###################################
# prediction, backtest & analysis
###################################
port_analysis_config = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "model": model,
            "dataset": dataset,
            "topk": 50,
            "n_drop": 5,
        },
    },
    "backtest": {
        "start_time": "2017-01-01",
        "end_time": "2020-08-01",
        "account": 100000000,
        "benchmark": benchmark,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": 0.095,
            "deal_price": "close",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5,
        },
    },
}

# backtest and analysis
with R.start(experiment_name="backtest_analysis"):
    recorder = R.get_recorder(recorder_id=rid, experiment_name="train_model")
    model = recorder.load_object("trained_model")

    # prediction
    recorder = R.get_recorder()
    ba_rid = recorder.id
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # backtest & analysis
    par = PortAnaRecord(recorder, port_analysis_config, "day")
    par.generate()


# 从实验记录器加载保存在pkl文件中的预测结果数据
pred_df = recorder.load_object("pred.pkl")

# 从实验记录器加载保存在pkl文件中的标签数据w 
label_df = recorder.load_object("label.pkl")
label_df.columns = ['label']

# # 构造预测值和标签值并列的df
# pred_label = pd.concat([pred_df, label_df], axis=1, sort=True).reindex(label_df.index)

# print(pred_label)
