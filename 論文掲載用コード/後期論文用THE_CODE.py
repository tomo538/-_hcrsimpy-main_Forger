import pylab as plt
import numpy as np
import time as time
import matplotlib as mpl
import copy as copy
mpl.use('Agg')  # 画面表示をしない設定

from HCRSimPY.plots import *
from HCRSimPY.light_schedules import *
from HCRSimPY.light_schedules import LuminusPreOptimised
from HCRSimPY.light_schedules import J_calc
from HCRSimPY.light_schedules import apply_optimized_intensities_to_schedule
from HCRSimPY.models import *
from HCRSimPY.plots import actogram
from tqdm import tqdm

def J_calc_threshold(opt, threshold):
    """コスト関数Jの個々のモデルを計算する"""
    #for i, model in enumerate(models):
        # 各モデルの結果から最終状態を取得
    higher_x    = threshold[0,0]
    higher_xc   = threshold[0,1]
    lower_x     = threshold[1,0]
    lower_xc    = threshold[1,1]
    # 目標値との誤差の二乗和を計算
    higher_squared_error = np.sqrt(pow((higher_x - opt[0]),2) + pow((higher_xc - opt[1]),2))
    lower_squared_error = np.sqrt(pow((lower_x - opt[0]),2) + pow((lower_xc - opt[1]),2))

    Jth = np.array([higher_squared_error, lower_squared_error])
    print(f"Jの閾値Jth = {Jth}") # デバッグ用に追加
    return (Jth)


whole_time_start = time.perf_counter() #処理時間計測、開始
#print関数で10000個まで要素を省かずに記述させる
np.set_printoptions(threshold=10000)
ts_pattern              = [7.5, 10, 12.5, 15, 20, 25, 30, 35]# 8種 大きい方がよい
min_lux_pattern         = [0, 15]# 2種　小さい方がよい
max_lux_pattern         = [1000, 10000]#, 10000]# 2種　どちらでもほとんど差はない
pre_optlux_pattern      = [1000, 500, 300, 250, 200, 150, 0]#,   7種 150付近が一番多い
optimize_days_pattern   = [7, 8, 9, 10, 11, 12]# 6種　




for e in range(len(optimize_days_pattern)):
    # 最小照度のループ
    optimize_days   = optimize_days_pattern[e]   # 最適化日数
    for d in range(len(pre_optlux_pattern)):
        # 最大照度のループ
        pre_optlux      = pre_optlux_pattern[d]
        
        # taux_diff = [23.8] # 他人のスケジュールを使うときの関数                    
        pre_period      = 10     # 準備期間の日数[days]
        dt              = 0.1   # 時間の最小
        dt_used         = 0.1   # シミュレーションで考慮される時間刻み
        sim_time        = 24*(pre_period + optimize_days + 1) # シミュレーション期間(月曜からスタート)
        #office_time     = [9, 18]   # 最適化を受ける時間帯はオフィス時間と仮定
        t0_idx          = int(24 * pre_period / dt) # 最適化開始のidx
        tf_idx          = int((24 * (pre_period + optimize_days)) / dt)   # 最適化終了のidx

        max_iterations  = 0     # 無限ループ防止
        J_slight_tremor = 0.00001   # Jの変化量が小さすぎる場合、ループ終了
        cost_add_allow  = 3     # コスト増加の許容回数

        max_move        = 50    # 最適化の移動距離を制限
        # 最初の体内時計のスケジュール用時間と、それに合わせた初期体内時計の位相と、目標の時間が必要。
        back_time       = 8    # 最適化開始時刻からの過去時間[hours]。opt_tの8時間前
        opt_t           = 24-back_time    # 準備中の時間帯決める？(opt_t時の位相を目標位相とする)
        Outside_lux     = [ 0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,7300,7300,8200,8000,8900,8000,7900,9400,
                            8300,7200,7000,6700,7500,6900,6300,6800,8400,6100,
                            6400,6800,8400,8900,10000,9300,8600,7600,7400,6800,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,10100,
                            10000,9900,10200,10500,10800,11200,11500,11800,11600,11400,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,1241,1174,1106,948 ,700 ,451 ,
                            414 ,378 ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,
                            0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0   ,0]
        train_lux       = 150      # 通勤時間帯の照明の明るさ
        dining_lux      = 200   # 食事時間帯の照明の明るさ
        office_lux      = 400   # オフィス時間帯の照明の明るさ
        taux            = [23.8, 24.1, 24.3] #体内時計の設定。個人差を表す。
        # 最適化する予定となる時間帯の初期化
        homelux     = [150, 150, 150]   # 家の照明の明るさ
        
        # 平日(仕事あり)の予定
                        # [モデル1, モデル2, モデル3]
        wake            = [6.5 , 8.0,  7.5]#, 7.5]    # 起床時刻[h]
        go_out_home     = [8.3 , 9.5, 10.0]#, 8]    # 外出時刻[h]
        
        into_office     = [9.7 , 10.5, 10.7]#, 10.5]    # 出社時刻[h]
        lunch           = [12.0, 12.0, 12.5]#, 12]    # 昼食離席[h]
        
        lunch_fin       = [13.0, 12.7, 13.0]#, 13]    # 昼食終了[h]
        leave_office    = [16.5, 18.0, 18.0]#, 17.5]    # 退社時刻[h]
        
        into_home       = [17.8, 19.0, 19.0]#, 18.5]    # 帰宅時刻[h]
        bedtime         = [23.5, 24.0, 24.5]    # 就寝時刻[h]

        # 休日の予定
        wake_holi       = [8.0 ,10.5, 9.5]    # 起床時刻[h]
        go_out_home_holi= [11.5,12.0,11.0]#, 10]    # 外出時刻[h]
        into_home_holi  = [12.5,20.0,13.0]#, 10]    # 帰宅時刻[h]
        Bedtime_holi    = [22.5,24.0,26.0]    # 就寝時刻[h]
        """
        
        taux            = [23.8]#, 24.1]#, 24.3] #体内時計の設定。個人差を表す。
        # 最適化する予定となる時間帯の初期化
        #homelux     = [150]#, 150]#, 150]   # 家の照明の明るさ
        #pre_optlux  = 200   # 最適化前の照明の明るさ
        # 平日(仕事あり)の予定
                        # [田中, 高橋, 川野輪]
        wake            = [6.5]# , 8.0]# , 7.5]#, 7.5]    # 起床時刻[h]
        go_out_home     = [8.3]# , 9.5]#, 10.0]#, 8]    # 外出時刻[h]
        into_office     = [9.7]# ,10.5]#, 10.7]#, 10.5]    # 出社時刻[h]
        lunch           = [12.0]#, 12.0]#, 12.5]#, 12]    # 昼食離席[h]
        lunch_fin       = [13.0]#, 12.7]#, 13.0]#, 13]    # 昼食終了[h]
        leave_office    = [16.5]#, 18.0]#, 18.0]#, 17.5]    # 退社時刻[h]
        into_home       = [17.8]#, 19.0]#, 19.0]#, 18.5]    # 帰宅時刻[h]
        bedtime         = [23.5]#, 24.0]#, 24.5]    # 就寝時刻[h]
        # 休日の予定
        wake_holi       = [8.0]# ,10.5]#, 9.5]    # 起床時刻[h]
        go_out_home_holi= [11.5]#,12.0]#,11.0]#, 10]    # 外出時刻[h]
        into_home_holi  = [12.5]#,20.0]#,13.0]#, 10]    # 帰宅時刻[h]
        Bedtime_holi    = [22.5]#,24.0]#,26.0]    # 就寝時刻[h]

        
        taux            = [24.1]#, 24.1, 24.3] #体内時計の設定。個人差を表す。
        # 最適化する予定となる時間帯の初期化
        homelux     = [150]#, 150, 150]   # 家の照明の明るさ
        pre_optlux  = 200   # 最適化前の照明の明るさ
        # 平日(仕事あり)の予定
                        # [高橋]
        wake            = [8.0]# , 8.0 , 7.5]#, 7.5]    # 起床時刻[h]
        go_out_home     = [9.5]# , 9.5, 10.0]#, 8]    # 外出時刻[h]
        into_office     = [10.5]# ,10.5, 10.7]#, 10.5]    # 出社時刻[h]
        lunch           = [12.0]#, 12.0, 12.5]#, 12]    # 昼食離席[h]
        lunch_fin       = [12.7]#, 12.7, 13.0]#, 13]    # 昼食終了[h]
        leave_office    = [18.0]#, 18.0, 18.0]#, 17.5]    # 退社時刻[h]
        into_home       = [19.0]#, 19.0, 19.0]#, 18.5]    # 帰宅時刻[h]
        bedtime         = [24.0]#, 24.0, 24.5]    # 就寝時刻[h]
        # 休日の予定
        wake_holi       = [10.5]# ,10.5, 9.5]    # 起床時刻[h]
        go_out_home_holi= [12.0]#,12.0,11.0]#, 10]    # 外出時刻[h]
        into_home_holi  = [20.0]#,20.0,13.0]#, 10]    # 帰宅時刻[h]
        Bedtime_holi    = [24.0]#,24.0,26.0]    # 就寝時刻[h]
        


        taux            = [24.3] #体内時計の設定。個人差を表す。
        # 最適化する予定となる時間帯の初期化
        homelux     = [150]   # 家の照明の明るさ
        pre_optlux  = 200   # 最適化前の照明の明るさ
        # 平日(仕事あり)の予定
                        # [川野輪]
        wake            = [7.5]    # 起床時刻[h]
        go_out_home     = [10.0]    # 外出時刻[h]
        into_office     = [10.7]    # 出社時刻[h]
        lunch           = [12.5]    # 昼食離席[h]
        lunch_fin       = [13.0]    # 昼食終了[h]
        leave_office    = [18.0]    # 退社時刻[h]
        into_home       = [19.0]    # 帰宅時刻[h]
        bedtime         = [24.5]    # 就寝時刻[h]
        # 休日の予定
        wake_holi       = [9.5]    # 起床時刻[h]
        go_out_home_holi= [11.0]    # 外出時刻[h]
        into_home_holi  = [13.0]    # 帰宅時刻[h]
        Bedtime_holi    = [26.0]    # 就寝時刻[h]


        taux            = [23.8, 24.1]#, 24.3] #体内時計の設定。個人差を表す。
        # 最適化する予定となる時間帯の初期化
        homelux     = [150, 150]#, 150]   # 家の照明の明るさ
        pre_optlux  = 200   # 最適化前の照明の明るさ
        # 平日(仕事あり)の予定
                        # [田中, 高橋, 川野輪]
        wake            = [6.5 , 8.0]# , 7.5]#, 7.5]    # 起床時刻[h]
        go_out_home     = [8.3 , 9.5]#, 10.0]#, 8]    # 外出時刻[h]
        into_office     = [9.7 ,10.5]#, 10.7]#, 10.5]    # 出社時刻[h]
        lunch           = [12.0, 12.0]#, 12.5]#, 12]    # 昼食離席[h]
        lunch_fin       = [13.0, 12.7]#, 13.0]#, 13]    # 昼食終了[h]
        leave_office    = [16.5, 18.0]#, 18.0]#, 17.5]    # 退社時刻[h]
        into_home       = [17.8, 19.0]#, 19.0]#, 18.5]    # 帰宅時刻[h]
        bedtime         = [23.5, 24.0]#, 24.5]    # 就寝時刻[h]
        # 休日の予定
        wake_holi       = [8.0 ,10.5]#, 9.5]    # 起床時刻[h]
        go_out_home_holi= [11.5,12.0]#,11.0]#, 10]    # 外出時刻[h]
        into_home_holi  = [12.5,20.0]#,13.0]#, 10]    # 帰宅時刻[h]
        Bedtime_holi    = [22.5,24.0]#,26.0]    # 就寝時刻[h]
        """
        
        # 後期変更部分ここから
        person_windows = []  # 各個人の最適化ウィンドウを格納するリスト
        for i in range(len(taux)):
            person_windows.append([
                    (wake           [i],   go_out_home [i]),
                    (into_office    [i],   lunch       [i]),
                    (lunch_fin      [i],   leave_office[i]),
                    (into_home      [i],   bedtime     [i])
                ])
        include_weekends = True

        # オフィス時間のみだった既存の生成を置換：
        # 朝 / オフィス / 夕 のウィンドウ（参加者全体の最小/最大）を作り、1日分の最適化候補を並べる
        morning_start = int(np.floor(np.min(wake)))         # floorは小数点以下の切り捨て
        morning_end   = int(np.ceil(np.max(go_out_home)))   # ceilは小数点以下の切り上げ
        office_start  = int(np.floor(np.min(into_office)))
        office_end    = int(np.ceil(np.max(leave_office)))
        evening_start = int(np.floor(np.min(into_home)))
        evening_end   = int(np.ceil(np.max(bedtime)))

        optimization_windows = [
            #(morning_start, morning_end),
            (office_start,  office_end),
            #(evening_start, evening_end)
        ]

        # 1日あたりの時間長（hours）
        worktime = sum(max(0, end - start) for (start, end) in optimization_windows)
        if worktime <= 0:
            raise ValueError("最適化ウィンドウの合計が0です。wake/office/bedtimeの設定を確認してください。")
        q = int(worktime * optimize_days / dt)  # 制御点数
        print("照明の制御回数:", q, "worktime(h/day):", worktime)
        
        l_j = np.zeros((q, 2), dtype=int)
        idx = 0
        for day in range(optimize_days):
            day_base = (pre_period + day) * 24.0
            for (start_h, end_h) in optimization_windows:
                # case 1: ウィンドウが当日内に収まる
                if end_h <= 24.0:
                    t = start_h
                    while t < end_h and idx < q:
                        abs_hour = day_base + t
                        l_j[idx, 0] = int(round(abs_hour / dt))
                        l_j[idx, 1] = int(pre_optlux)
                        idx += 1
                        t += dt_used
                else:
                    # ウィンドウが日をまたぐ (例 end_h=26)
                    # 当日の分: start_h .. 24.0
                    t = start_h
                    while t < 24.0 and idx < q:
                        abs_hour = day_base + t
                        l_j[idx, 0] = int(round(abs_hour / dt))
                        l_j[idx, 1] = int(pre_optlux)
                        idx += 1
                        t += dt_used
                    # 翌日の早朝分: 0 .. (end_h - 24.0)
                    morning_end = end_h - 24.0
                    if (day + 1) < optimize_days:
                        t2 = 0.0
                        next_day_base = (pre_period + day + 1) * 24.0
                        while t2 < morning_end and idx < q:
                            abs_hour = next_day_base + t2
                            l_j[idx, 0] = int(round(abs_hour / dt))
                            l_j[idx, 1] = int(pre_optlux)
                            idx += 1
                            t2 += dt_used
                if idx >= q:
                    break
            if idx >= q:
                break
        # safety 填充（dt基準のインデックス）
        if idx < q:
            fill_idx = int(round((pre_period * 24.0) / dt))
            for k in range(idx, q):
                l_j[k, 0] = fill_idx
                l_j[k, 1] = int(pre_optlux)

        process_time_start = time.perf_counter() #処理時間計測、開始

        models = []     # 体内時計の個数分の個人インスタンスを作成
        Jth = np.zeros((len(taux), 2), dtype=float)    # Jの目標位相との差の目標
        bodyclock_init = []       # 初期位相を格納
        bodyclock_opt = []        # 目標とする位相を格納[-0.922, 0.492, 0.037],[-1.026, 0.315, 0.037],[-1.084, 0.145, 0.037]    [23.8,24.0,24.2]の並び順　[1,1,0]の約8時間後の位相
        bodyclock_threshold = []   # 目標位相の閾値を格納
        base_schedules = []
        stable_initial_state = [[], []] # 過渡状態からの初期値を格納する
        for i in range(len(taux)):
            # 各個人のインスタンスを生成。初期状態を設定する。
            model = vdp_forger99_model(taux[i], sim_time, dt)       #各個人のインスタンス生成。パラメーターが設定される。
            
            transient_schedule = np.array([LuminusPreOptimised_initial(
                t, homelux[i], pre_optlux, t0_idx, opt_t, dt,
                Outside_lux, train_lux, dining_lux,
                wake[i], go_out_home[i], into_office[i], lunch[i], lunch_fin[i], leave_office[i], into_home[i], bedtime[i],
                wake_holi[i], go_out_home_holi[i], into_home_holi[i], Bedtime_holi[i]
            ) for t in model.ts])
            model.light_schedule = transient_schedule
            stable_state = model.integrateTransients(back_time, numdays=50)

            bodyclock_init.append(stable_state[0])   # 過渡状態から、初期位相(0時の位相)を取得
            bodyclock_opt.append(stable_state[1]) # 過渡状態から、目標位相(opt_t時の位相)を取得
            bodyclock_threshold.append(stable_state[2]) # 目標位相の閾値を取得
            
            Jth[i, :] = J_calc_threshold(bodyclock_opt[i], bodyclock_threshold[i]) # 各モデルのJ計算用の閾値を設定する

            #　ここから自由なスケジュールに書き変える
            # 初期スケジュールを作成
            base_schedule = np.array([LuminusPreOptimised(t, homelux[i], pre_optlux, t0_idx, opt_t, dt, 
                Outside_lux, train_lux, dining_lux, office_lux, 
                wake[i], go_out_home[i], into_office[i], lunch[i], lunch_fin[i], leave_office[i], into_home[i], bedtime[i], 
                wake_holi[i], go_out_home_holi[i], into_home_holi[i], Bedtime_holi[i]) 
                for t in model.ts
            ])
            base_schedules.append(base_schedule) # スケジュールを入れる
            initial_schedule = apply_optimized_intensities_to_schedule(
                base_schedules[i], l_j, dt, dt_used, lunch_times=[lunch[i], lunch_fin[i]],
                office_times=[into_office[i], leave_office[i]], pre_period=pre_period,
                optimize_days=optimize_days, person_windows=person_windows[i], include_weekends=include_weekends) # 最適化前のスケジュールを作成
            model.light_schedule = initial_schedule        # 初期スケジュールを設定
            model.integrateModel(sim_time, bodyclock_init[i]) #引数は24時間 × n日ぶんの意。モデルを時間単位ごとに順方向積分し、モデルに入れ、位相を計算する。
            models.append(model)                                    # インスタンスをmodels配列に格納
            #print("初期位相 = ", initial) # 初期位相を表示
            #print("目標位相 = ", opt[i])

        # 初期値や目標値のコピーを保存
        models_copy         = copy.deepcopy(models)
        base_schedules_copy = copy.deepcopy(base_schedules)
        l_j_copy            = copy.deepcopy(l_j)

        process_time_end = time.perf_counter() #処理時間計測、終了
        print('光スケ初期化処理時間：{:.3f}'.format((process_time_end-process_time_start))) #分単位で表示
        J = []                                  # コスト関数
        initial_J, initial_Each_J = J_calc(models, bodyclock_opt, tf_idx, Jth) # Jの初期値を求める
        J.append(initial_J)                     # Jに入れる


        for c in range(len(max_lux_pattern)):
        # 最適化前照度のループ
            max_lux = max_lux_pattern[c]
            for b in range(len(min_lux_pattern)):
                min_lux = min_lux_pattern[b]
                for a in range(len(ts_pattern)): 
                    ts = ts_pattern[a]    # ステップサイズ
                    print(f"--- {e+1}/{len(optimize_days_pattern)}番目の最適化日数: {optimize_days_pattern[e]} ---")
                    print(f"--- {d+1}/{len(pre_optlux_pattern)}番目の最適化前照度: {pre_optlux_pattern[d]} ---")
                    print(f"--- {c+1}/{len(max_lux_pattern)}番目の最大照度: {max_lux_pattern[c]} ---")
                    print(f"--- {b+1}/{len(min_lux_pattern)}番目の最小照度: {min_lux_pattern[b]} ---")
                    print(f"--- {a+1}/{len(ts_pattern)}番目のステップサイズ: {ts_pattern[a]} ---")

                    # 各変数をコピーで初期化
                    # models_copy はテンプレのスナップショット（deepcopy済）なので、
                    # ここでさらに deepcopy して「独立した復元版」を作る
                    models = copy.deepcopy(models_copy)
                    base_schedules = copy.deepcopy(base_schedules_copy)
                    l_j = copy.deepcopy(l_j_copy)
                    cost_add_num    = 0     # コストの増加回数
                    goal_complete  = False  # 目標達成フラグ
                    J = [initial_J]  # Jの初期化

                    # ここから最適化処理のstep2～7のループ
                    for iteration in range(max_iterations): # このループで一周(q個ぶんのl_j)回すこれを条件満たすまで繰り返す。
                        print(f"--- Iteration {iteration + 1} ---")
                        print(f"ts:{a+1}/{len(ts_pattern)},    min:{b+1}/{len(min_lux_pattern)},    max:{c+1}/{len(max_lux_pattern)},    pre_optlux:{d+1}/{len(pre_optlux_pattern)},    opt_days:{e+1}/{len(optimize_days_pattern)}")
                        """
                        step2: スイッチング時間を用いて、システム方程式をx0から順方向に積分し、状態軌跡x(t)を決定します。
                        """
                        process_time_start = time.perf_counter() #処理時間計測、開始
                        for i,model in enumerate(models):
                            current_schedule = apply_optimized_intensities_to_schedule(
                                base_schedules[i], l_j, dt, dt_used, lunch_times=[lunch[i], lunch_fin[i]],
                                office_times=[into_office[i], leave_office[i]],
                                pre_period=pre_period, optimize_days=optimize_days,
                                person_windows=person_windows[i], include_weekends=include_weekends)
                            model.light_schedule = current_schedule
                            model.recalculate_dynamics(t0_idx, tf_idx)
                        process_time_end = time.perf_counter() #処理時間計測、終了
                        print('step2処理時間：{:.3f}'.format((process_time_end-process_time_start))) #分単位で表示

                        # step7
                        current_J, Each_J = J_calc(models, bodyclock_opt, tf_idx, Jth)
                        J.append(current_J) # Jにj+1番目のJであるtotalを追加
                        print(f"Current Cost J = {current_J}")

                        if J[-1] > J[-2]:
                            cost_add_num += 1
                            if cost_add_num > cost_add_allow: # コスト増加時は終了
                                print("コストが増加したため、前のステップで終了します。")
                                break
                        goal_achieved_num = 0
                        for i, model in enumerate(models):
                            if model.goal_achieved == True:
                                goal_achieved_num += 1
                        if goal_achieved_num == len(models): # 全員目標達成したら終了
                            goal_complete = True
                            print("収束しました。")
                            break
                        if (0 <= J[-2] - J[-1] < J_slight_tremor): #　ΔJ<0.00001のとき終了
                            cost_add_num += 1
                            if cost_add_num > cost_add_allow:
                                print("Jの変化量が小さすぎるため、前のステップで終了します。")
                                break

                        """
                        step3 共状態方程式をtfから逆方向に積分し、スイッチング時間におけるλTを決定します。
                        """
                        process_time_start = time.perf_counter() #処理時間計測、開始
                        for i,model in enumerate(models):# 各インスタンスに操作するためのループ
                            # step3
                            model.calculate_adjoint_dynamics(bodyclock_opt[i], tf_idx, t0_idx)
                        process_time_end = time.perf_counter() #処理時間計測、終了
                        print('step3処理時間：{:.3f}'.format((process_time_end-process_time_start))) #分単位で表示

                        """
                        step4 式に従ってコストJを減少させる最適な摂動(o˙u)を計算します。
                        """
                        process_time_start = time.perf_counter() #処理時間計測、開始
                        sens_tf_list = [] # 各モデルの感度を格納するリスト
                        # 後期の改善点ここから
                        # o・u = dJ/dI_tf　での正規化？を追加
                        for model in models: # 各モデルに対して、dl_jを計算する
                            sens_tf_list.append(abs(model.calculate_sensitivity_for_intensity_tf(tf_idx)))
                                
                        dl_j_array = np.zeros(q)# このイテレーションで使うd_tの微小移動dl_jを格納する配列
                        for j in range(q):
                            start_idx = int(l_j[j, 0])
                            end_idx = start_idx + int(1/dt)
                            sens_list = [] # 各モデルの感度を格納するリスト
                            for model in models: # 各モデルに対して、dl_jを計算する
                                sens_list.append((model.calculate_sensitivity_for_intensity(start_idx, end_idx))/sens_tf_list[models.index(model)]) # 各モデルの感度を正規化してリストに追加
                            # 後期の改善点ここまで
                            #print(f"sens_list :{sens_list}")
                            dl_j_array[j] = np.mean(sens_list)              # 時刻l_j[j～j+1]のdJ/dIを入れる
                            
                            #for i in range(len(dl_j_list)): # 重みづけを変えたいとき用
                            #avg_dl_j = 0
                            #avg_dl_j += (dl_j_list[0]*0.6) + (dl_j_list[1]*0.25) + (dl_j_list[2]*0.15)
                            # モデル間の平均を取って、照度の移動量を決定
                            # 一周分の微動値が入る
                        process_time_end = time.perf_counter() #処理時間計測、終了
                        print('step4処理時間：{:.3f}'.format((process_time_end-process_time_start))) #分単位で表示

                        #print(f"dl_j_array :{dl_j_array}")
                        """
                        step5: スイッチング時間を調整する
                        """
                        process_time_start = time.perf_counter() #処理時間計測、開始
                        move_amount = (ts * dl_j_array).astype(int) # dl_jを整数の値にする。
                        #print(f"  move amount: {move_amount}")           # 表示
                        clipped_move_amount = np.clip(move_amount, -max_move, max_move)  # np.clip(配列, 最小値, 最大値) で範囲内に収める
                        final_move = clipped_move_amount.astype(int)  # 整数に変換
                        #print(f"  Final move amount: {final_move}")  # デバッグ用に追加
                        l_j[:, 1] += final_move  # l_jが最適化された時間の番号を指すように更新する
                        l_j[:, 1] = np.clip(l_j[:, 1], min_lux, max_lux)  # 明るさの上下限制限
                        process_time_end = time.perf_counter()  # 処理時間計測、終了
                        print('step5処理時間：{:.3f}'.format((process_time_end-process_time_start)))  # 分単位で表示
                    #print('light_schedule =' ,models[0].light_schedule)

                    # 他人のスケジュールを使う際のコード
                    #for i in range(len(taux_diff)):
                    #    modelz = vdp_forger99_model(taux_diff[i], sim_time, dt)#各個人のインスタンス生成。パラメーターが設定される。
                    #    modelz.integrateModel(sim_time, initial = [1.0, 1.0, 0])#引数は24時間 × n日ぶんの意。モデルを時間単位ごとに順方向積分し、モデルに入れ、位相を計算する。
                    #    models.append(modelz)
                    for i,model in enumerate(models):
                        model.finalcalc(t0_idx, tf_idx)
                        current_J, Each_J = J_calc(models, bodyclock_opt, tf_idx, Jth)
                        J.append(current_J) # Jにj+1番目のJであるtotalを追加
                        print(f"Current Cost J = {current_J}")


                    # --- 最適化後の最終コストを取得 ---
                    final_J = J[-1]
                    
                    process_time_start = time.perf_counter() #処理時間計測、開始
                    # 以上までで最適化スケジュールが完成するので、そのスケジュールを用いた作図を↓と同様に行う。
                    for i, model in enumerate(models): # ★ enumerateを使って、モデルのインデックスも取得
                        tsdf = model.getTS() # Time、Light_Level (lux)、Phase (radians)、R (振幅)、n (光活性化変数) を列とするpandasデータフレーム取得する。
                        
                        #以下は図(アクトグラム)を描くためのコード
                        fig = plt.figure(figsize=(6.4, 4.8)) # ★ figオブジェクトを取得するために、plt.figure()を変数に代入
                        ax = plt.gca()
                        start_day_marker = pre_period 
                        end_day_marker = pre_period + optimize_days - 1

                        acto = actogram(ax, tsdf, 
                                        opt_start_day=start_day_marker, 
                                        opt_end_day=end_day_marker
                                        )

                        # --- ★★★ ここからが追加・修正部分 ★★★ ---
                        
                        # グラフのタイトルを設定
                        plt.title(f'Optimized Schedule for Model {i+1} (taux={model.taux})')
                        
                        # fig.text()を使って、グラフの下部にJの値を書き込む
                        # Jの値が数値の場合のみフォーマットする
                        if isinstance(final_J, (int, float)):
                            j_text = f"Final Cost J = {final_J:.4f}"#, Each J = {Each_J[i]:.4f}"
                        else:
                            j_text = f"Final Cost J = {final_J}"

                        fig.text(
                            0.5,                             # x座標: 0.5 (図の水平方向の中央)
                            0,                               # y座標: 0 (図の非常に下の方)
                            j_text,                          # 表示するテキスト
                            ha='center',                     # 水平方向の揃え: 中央揃え
                            va='bottom',                     # 垂直方向の揃え: 下揃え
                            fontsize=12,
                            color='dimgray'
                        )
                        
                        # レイアウトの調整
                        # tight_layout()の代わりに、subplots_adjust()で手動調整する方が確実
                        plt.tight_layout() # 上下のマージンを調整
                        # --- ★★★ 追加・修正ここまで ★★★ ---
                        
                        
                        # plt.show() は、すべてのグラフの生成が終わった後に一度だけ呼び出すのが一般的
                        fig.savefig(f'論文用/モデル{i+1}_全員達成{goal_complete}_目標達成{model.goal_achieved}_最低照度{min_lux}_最大照度{max_lux}_最適化前照度{pre_optlux}_最適化日数{optimize_days}_ステップサイズ{ts}_J{final_J:.4f}.svg')#_Each_J{Each_J[i]:.4f}.svg') # グラフを保存する
                        #plt.show()
                        plt.close(fig) # 各図を閉じる。メモリを節約するため。
                    process_time_end = time.perf_counter() #処理時間計測、終了
                    print('表作成処理時間：{:.3f}'.format((process_time_end-process_time_start))) #分単位で表示



whole_time_end = time.perf_counter() #処理時間計測、終了
print('全体処理時間：{:.3f}'.format((whole_time_end-whole_time_start))) #分単位で表示
# --- ここから追加: 実行時間をログファイルに保存 ---
from datetime import datetime
log_line = f"{datetime.now().isoformat()} total_seconds={(whole_time_end-whole_time_start):.3f}\n"
with open("run_times.log", "a") as lf:
    lf.write(log_line)
# --- ここまで追加 ---



