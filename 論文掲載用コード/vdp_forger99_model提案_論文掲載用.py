
"""
このファイルはForger 1999モデルを定義しています。

Forger, D. B., Jewett, M. E., & Kronauer, R. E. (1999).
A Simpler Model of the Human Circadian Pacemaker. Journal of Biological Rhythms,
14(6), 533–538. https://doi.org/10.1177/074873099129000867

ただし、以下の論文から取得したパラメータを使用しています。

Serkh K, Forger DB. Optimal schedules of
light exposure for rapidly correcting circadian misalignment.
PLoS Comput Biol. 2014;10(4):e1003523. Published 2014 Apr 10. doi:10.1371/journal.pcbi.1003523

これは、原論文のパラメータとは異なります。
"""

from HCRSimPY.light_schedules import *
import scipy.integrate as sp_integrate
from scipy.interpolate import interp1d


"""     ---効率化用コードここから---     """
# ...existing code...
from numba import njit
import numpy as np
# ...existing code...

# --- numba/RK4 固定刻み積分（順方向） ---
@njit(cache=True)
def _derv_numba(x, xc, n, light_val, mu, G, alpha_0, delta, p, I0, kparam, taux):
    if light_val <= 0.0:
        current_alpha = 0.0
    else:
        current_alpha = alpha_0 * (light_val / I0) ** p

    B_hat = G * (1.0 - n) * current_alpha
    B = B_hat * (1.0 - 0.4 * x) * (1.0 - 0.4 * xc)

    dx = (np.pi / 12.0) * (xc + B)
    dxc = (np.pi / 12.0) * (mu * (xc - 4.0 / 3.0 * (xc ** 3)) - x * ((24.0 / (0.99669 * taux)) ** 2 + kparam * B))
    dn = 60.0 * (current_alpha * (1.0 - n) - delta * n)
    return dx, dxc, dn


@njit(cache=True)
def integrate_forward_numba(results, light_schedule, start_idx, end_idx, dt,
                            mu, G, alpha_0, delta, p, I0, kparam, taux):
    """
    results: (N,3) numpy array, light_schedule: (N,) numpy array
    start_idx..end_idx are indices (inclusive end_idx)
    results[start_idx] must contain initial state
    """
    n_rows = results.shape[0]
    if end_idx >= n_rows:
        end_idx = n_rows - 1
    for idx in range(start_idx, end_idx):
        x0 = results[idx, 0]
        xc0 = results[idx, 1]
        n0 = results[idx, 2]

        I = light_schedule[idx] if idx < light_schedule.shape[0] else 0.0

        # k1
        k1x, k1xc, k1n = _derv_numba(x0, xc0, n0, I, mu, G, alpha_0, delta, p, I0, kparam, taux)

        # k2
        x_k2 = x0 + 0.5 * dt * k1x
        xc_k2 = xc0 + 0.5 * dt * k1xc
        n_k2 = n0 + 0.5 * dt * k1n
        k2x, k2xc, k2n = _derv_numba(x_k2, xc_k2, n_k2, I, mu, G, alpha_0, delta, p, I0, kparam, taux)

        # k3
        x_k3 = x0 + 0.5 * dt * k2x
        xc_k3 = xc0 + 0.5 * dt * k2xc
        n_k3 = n0 + 0.5 * dt * k2n
        k3x, k3xc, k3n = _derv_numba(x_k3, xc_k3, n_k3, I, mu, G, alpha_0, delta, p, I0, kparam, taux)

        # k4
        x_k4 = x0 + dt * k3x
        xc_k4 = xc0 + dt * k3xc
        n_k4 = n0 + dt * k3n
        k4x, k4xc, k4n = _derv_numba(x_k4, xc_k4, n_k4, I, mu, G, alpha_0, delta, p, I0, kparam, taux)

        x_next = x0 + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        xc_next = xc0 + (dt / 6.0) * (k1xc + 2.0 * k2xc + 2.0 * k3xc + k4xc)
        n_next = n0 + (dt / 6.0) * (k1n + 2.0 * k2n + 2.0 * k3n + k4n)

        if idx + 1 < n_rows:
            results[idx + 1, 0] = x_next
            results[idx + 1, 1] = xc_next
            results[idx + 1, 2] = n_next
    # --- numba/RK4 部分ここまで ---

# --- ここから step3 (随伴方程式) 高速化用コード ---
@njit(cache=True)
def _compute_jacobians_njit(results, light_schedule, dt, mu, G, alpha_0, delta, p, I0, kparam, taux):
    n = results.shape[0]
    jac = np.zeros((n, 3, 3))
    pi_over_12 = np.pi / 12.0
    C1 = (24.0 / (0.99669 * taux)) ** 2.0

    for i in range(n):
        x, xc, n_var = results[i]
        current_light = light_schedule[i]
        
        if current_light > 1e-9:
            current_alpha = alpha_0 * (current_light / I0) ** p
        else:
            current_alpha = 0.0
            
        B_hat = G * (1.0 - n_var) * current_alpha
        B = B_hat * (1.0 - 0.4 * x) * (1.0 - 0.4 * xc)
        
        dBdx = B_hat * (-0.4) * (1.0 - 0.4 * xc)
        dBdxc = B_hat * (1.0 - 0.4 * x) * (-0.4)
        dBdn = G * (-1.0) * current_alpha * (1.0 - 0.4 * x) * (1.0 - 0.4 * xc)

        jac[i, 0, 0] = pi_over_12 * dBdx
        jac[i, 0, 1] = pi_over_12 * (1.0 + dBdxc)
        jac[i, 0, 2] = pi_over_12 * dBdn
        
        jac[i, 1, 0] = -pi_over_12 * (C1 + (kparam * x * dBdx) + (kparam * B))
        jac[i, 1, 1] = pi_over_12 * (mu * (1.0 - 4.0 * xc * xc) - kparam * x * dBdxc)
        jac[i, 1, 2] = -pi_over_12 * (kparam * x * dBdn)
        
        jac[i, 2, 2] = -60.0 * (current_alpha + delta)
        
    return jac

@njit(cache=True)
def _integrate_adjoint_backward_euler_njit(lambda_out, jacobians, start_idx, end_idx, dt):
    for idx in range(end_idx, start_idx, -1):
        Jt = jacobians[idx]
        lam = lambda_out[idx]
        
        v0 = -(Jt[0,0]*lam[0] + Jt[1,0]*lam[1] + Jt[2,0]*lam[2])
        v1 = -(Jt[0,1]*lam[0] + Jt[1,1]*lam[1] + Jt[2,1]*lam[2])
        v2 = -(Jt[0,2]*lam[0] + Jt[1,2]*lam[1] + Jt[2,2]*lam[2])
        
        lambda_out[idx-1, 0] = lam[0] + dt * v0
        lambda_out[idx-1, 1] = lam[1] + dt * v1
        lambda_out[idx-1, 2] = lam[2] + dt * v2
# --- step3 高速化用コードここまで ---

    """     ---効率化用コードここまで---     """


class vdp_forger99_model(object):
    """VDPベースモデル「Simpler Model」の実装"""

    def __init__(self, taux, tend, dt): #taux戻すselfはインスタンス自身を指し、LightFunはこのクラスに渡されたものを指す。
        
        """
        デフォルトのモデルパラメータを指定します。

        setParameters()

        戻り値: None
        """

        # パラメータを設定します λ=60、b=0.4、f=0.99669の定義は無し。
        self.taux = taux    #体内時計の長さ[h](周期)
        self.mu = 0.23
        self.G = 33.75
        self.alpha_0 = 0.05
        self.delta = 0.0075     #βとdeltaを間違えている？
        self.p = 0.50
        self.I0 = 9500.0
        self.kparam = 0.55

        # 用いるものを宣言する
        self.dt = dt
        self.ts = np.arange(0.0, tend + dt, dt) # 0.0からtendまでの時間ステップ配列
        self.light_schedule = np.zeros(len(self.ts))
        self.results = np.zeros((len(self.ts), 3))
        self.lambda_t = np.zeros((len(self.ts), 3)) # λT(t)を指す
        self.goal_achieved = False # 目標達成フラグ

    #def set_initial_light_schedule(self, LightFun,i):
    #    """初期の光スケジュールを関数から生成する"""
    #    self.light_schedule = np.array([LightFun(t) for t in self.ts])

    def set_initial_diff_light_schedule(self, diff_light_schedule):
        """初期の光スケジュールを関数から生成する"""
        self.light_schedule = diff_light_schedule


    def updateParameters(self, paramDict):
        """
        渡されたパラメータ辞書を使用してモデルパラメータを更新します。
        辞書に含まれていないパラメータは、デフォルト値に設定されます。

        updateParameters(paramDict)

        戻り値はnullで、クラスインスタンスに格納されているパラメータが変更されます。
        """

        params = ['taux', 'mu', 'G', 'alpha_0', 'delta', 'p', 'I0', 'kparam']

        # パラメータを設定します
        for k in paramDict.keys():
            mycode = 'self.' + k + "=paramDict[\'" + k + "\']"
            exec(mycode)

    def getParameters(self):
        """モデルオブジェクトで使用されている現在のパラメータの辞書を取得します。

        getParameters()

        パラメータの辞書を返します。
        """

        current_params = {
            'taux': self.taux,
            'mu': self.mu,
            'G': self.G,
            'alpha_0': self.alpha_0,
            'delta': self.delta,        
            'p': self.p,
            'I0': self.I0,
            'kparam': self.kparam}

        return (current_params)

    def alpha0(self, light_intensity):
        """光入力処理をモデル化するためのヘルパー関数です。"""
        # ここで照明スケジュールがLight(t)で考慮される
        if light_intensity <= 0:
            return 0.0
        return (self.alpha_0 * pow((light_intensity / self.I0), self.p))


    def derv(self, t, y):
        """
        この関数は、単一集団モデルのODEシステムを定義します。

        derv(self,t,y)

        dydt numpy配列を返します。
        """
        x = y[0]
        xc= y[1]
        n = y[2]

        idx = int(round(t / self.dt))
        if idx >= len(self.light_schedule):
            idx = len(self.light_schedule) - 1
        current_light = self.light_schedule[idx]
        # Bの計算で、alpha0に関数ではなく光強度を渡す
        # derv メソッド内

        current_alpha = self.alpha0(current_light)

        # 論文の定義に合わせる
        B_hat = self.G * (1.0 - n) * current_alpha
        B = B_hat * (1 - 0.4 * x) * (1 - 0.4 * xc)

        dydt = np.zeros(3)
        dydt[0] = np.pi / 12.0 * (xc + B) # BhatではなくB
        dydt[1] = np.pi / 12.0 * (self.mu * (xc - 4.0 / 3.0 * pow(xc, 3.0)) - x * (
            pow(24.0 / (0.99669 * self.taux), 2.0) + self.kparam * B)) # BhatではなくB
        dydt[2] = 60.0 * (current_alpha * (1.0 - n) - self.delta * n)
        return dydt


    def derv_adjoint(self, t, lambda_val, x, xc, n, light):
        """
        随伴(adjoint)方程式の右辺を計算します。
        lambda_val: 現在の随伴変数の値 [lambda_x, lambda_xc, lambda_n]
        t: 現在の時間 (逆方向積分の場合は、通常の時間軸での値)
        x, xc, n: 順方向積分で得られた状態変数 (x(t), xc(t), n(t)) の軌跡
                                これは補間して使用する必要があります。
        """
        # 時間 t における状態変数を補間して取得
        # 注意: solve_ivp の t は単一の時刻値なので、xなどは補間関数である必要があります
        # 例: interp1d を使うか、self.results から適切なインデックスを見つける
        # ここでは簡単のため、最も近いインデックスidx_tを使用すると仮定します。
        # 正確には、t_eval の範囲と一致するように self.results を使うべきです。
        idx_t = int(t / self.dt) # t が self.ts のインデックスに対応すると仮定
        if idx_t >= len(x) or idx_t < 0:
            # 範囲外の場合はエラー処理または適切な値を返す
            # 実際には、t_eval の範囲を t_j[j+1] から t_j[j] に設定するため、このチェックは重要です。
            return np.zeros(3) # あるいは例外を発生させる

        x = x[idx_t]
        xc = xc[idx_t]
        n = n[idx_t]
        intensity_at_t = light(t)
        current_alpha_I = self.alpha0(intensity_at_t)

        # df/dx の計算 (ヤコビ行列)
        # これらの値は、step3のdfdxの計算と同じロジックで計算されますが、
        # ここでは現在の時刻tと状態変数x, xc, nに基づいて計算されます。
        # self.dfdxは特定のt_jに対するものなので、ここでは動的に計算する必要があります。
        # あるいは、step2でself.dfdx_all_time = (tend,3,3)のように全時間で計算し保存しておく。
        # ここでは簡単化のため、dfdxの定義がクラス内にヘルパー関数としてあると仮定。
        # 例: self._compute_jacobian(x, xc, n) のような関数を用意する

        dfdx = np.zeros((3, 3))
        dfdx[0, 0] = (np.pi / 12.0) * self.dBdx(xc, n)
        dfdx[0, 1] = (np.pi / 12.0) * (1 + self.dBdxc(x, n))
        dfdx[0, 2] = (np.pi / 12.0) * (self.dBdn(x, xc))
        dfdx[1, 0] = (np.pi / 12.0) * ((24 / (0.99669 * self.taux))**2 - self.kparam * self.dBdx(xc, n))
        dfdx[1, 1] = -(np.pi / 12.0) * (self.mu * (1 - 4.0 * xc**2) - self.kparam * self.dBdxc(x, n))
        dfdx[1, 2] = (np.pi / 12.0) * self.kparam * self.dBdn(x, xc)
        #dfdx[2, 0] = 0 # 仮定
        #dfdx[2, 1] = 0 # 仮定
        dfdx[2, 2] = -60.0 * (self.alpha0(idx_t) + self.delta) # alpha0_func は時間によって変わるalpha0(I)の値を与える関数と仮定

        # 随伴方程式: dλ/dt = - (∂f/∂x)^T λ
        dlambda_dt = -np.dot(dfdx.T, lambda_val)
        return (dlambda_dt)


    def dBdx(self, xc, n):
        """dBdf計算をやりやすくしてる"""
        return (self.G * (1-n) * (-0.4) * (1 - 0.4*xc))
    
    def dBdxc(self, x, n):
        """dBdn計算をやりやすくしてる"""
        return (self.G * (1-n) * (1 - 0.4*x) * (-0.4))
    
    def dBdn(self, x, xc):
        """dBdn計算をやりやすくしてる"""
        return (self.G * (-1) * (1 - 0.4*x) * (1-0.4*xc))
    
    def dalphadu(self, t_j):
        """dαdu計算をやりやすくしてる"""
        return (self.alpha_0 * self.p * self.Light[t_j]**(self.p-1) / self.I0**self.p)

    def dBdu(self, x, xc, n, t_j):
        """dBdu計算をやりやすくしてる"""
        return (self.G * (1.0 - n) * (1 - 0.4 * x) * (1 - 0.4 * xc) * self.dalphadu(t_j))
    

    def integrateModel(self, tend, initial):
        """モデルを時間的に順方向積分します。

        integrateModel(tend, initial=[1.0, 0.0, 0.0])

        tend: 積分を停止する最終時間を表すfloatです。
        initial: 初期動的状態です。
        パラメータは、tend=シミュレーションを停止する終了時間、initial=[x, xc, n]です。

        積分結果をscipy配列self.resultsに書き込みます。

        システムの終了時刻における概日位相（時間単位）を返します。
        """

        # 常微分はやはり順次微分した結果を反映する処理らしい
        
        #   sp.integrate.solve_ivp(解く微分方程式,　積分を行う時間範囲,　状態変数の初期条件を示す配列,　解を評価する特定の時間点, 使用する積分アルゴリズム)
        r = sp.integrate.solve_ivp( self.derv   , (0, tend)        , initial                   , t_eval=self.ts         , method='Radau')  # RK45法を使用。内部で精度を保証できるよう時間刻みを勝手に調整してくれる。
        # rにdydt=x(t)が入ってる　r.yは各状態変数の時間発展を含む2次元配列
        """rの中身の例
        r.y = np.array([
        # x の値
        [1.000, 1.050, 1.090],  #  xの値、時刻0.0, 0.1, 0.2 における
        # xc の値
        [1.000, 0.980, 0.950],  #  xcの値、時刻0.0, 0.1, 0.2 における
        # n の値
        [0.000, 0.010, 0.020]   #  nの値、時刻0.0, 0.1, 0.2 における
        ])
        """
        
        if r.success:
            self.results = np.transpose(r.y)    # 結果の行列を転置してself.resultsに保存
        else:
            print(f"Warning: Integration failed for taux={self.taux}")
        return

    def integrateModelData(self, timespan, initial):
        """
        データによって定義された光関数を使用してモデルを積分します。

        integrateModelData(timespan, initial, dt=0.1)

        timespanは、開始時間と終了時間のタプルです（例：(0.0, 10.0)）。
        initialは、動的システムの初期条件です。
        dtは、scipyにシステムの動的状態を保存する頻度を伝えます。

        結果をnumpy配列self.resultsに書き込みます。
        """
        dt = 0.01
        self.ts = np.arange(timespan[0], timespan[1], dt)
        r = sp.integrate.solve_ivp(
            self.derv, (timespan[0], timespan[-1]), initial, t_eval=self.ts, method='Radau')
        self.results = np.transpose(r.y)

    def integrateTransients(self, back_time, numdays=50):
        """
        過渡現象を取り除くためにモデルをnumdays日間積分し、
        初期条件として使用する終点を返します。

        integrateTransients(numdays=50)

        モデルの最終状態を示すnumpy配列を返します。
        """
        tend = numdays * 24.0
        t_start_eval = 0.0  # 過渡現象を取り除くための積分開始時間
        num_eval_points = int(tend / self.dt) + 1 # 評価する時間点の数
        t_eval_points = np.linspace(t_start_eval, tend, num_eval_points) # 評価する時間点の配列
        #print(f"t_eval_pointsの長さ: {len(t_eval_points)}")
        #   sp.integrate.solve_ivp(解く微分方程式,　積分を行う時間範囲,　状態変数の初期条件を示す配列,　解を評価する特定の時間点, 使用する積分アルゴリズム)
        r = sp.integrate.solve_ivp(self.derv, (0, tend), [1.0, 1.0, 0.0], t_eval=t_eval_points, method='Radau')
        
        if not r.success:
            print("Warning: Transient integration failed.")
            # エラー時はダミーの値を返すなど、適切な処理が必要
            return (np.zeros(3), np.zeros(3))
        
        results_trans = np.transpose(r.y)
        #print(f"results_trans: {results_trans[-960:-1, 2]}")

        # --- 返り値を計算 ---
        # 最終状態 (tendの時点)
        final_state = 0
        target_state= 0
        threshold   = np.zeros((2,3),dtype=float) # 目標状態の前後7ステップを閾値として保存
        
        """
        # 配列の長さを超えないかチェック
        if abs(target_idx) > len(results_trans):
            print("Warning: opt_t is larger than the evaluated range in integrateTransients.")
            # エラー処理
            target_state = np.zeros(3)
        else:
            final_state = results_trans[-1, :]
            target_state = results_trans[target_idx, :]
        """
        Adjusting_each_body_clock_ini = 16.0 - (24.1-self.taux)*5 # 根拠のない調整用の数字。体内時計毎に異なるCBTminの時間に対処
        #target_idx += int((24.1-self.taux)*3/self.dt) # 根拠のない調整用の数字。体内時計毎に異なるCBTminの時間に対処
        final_i = 0

        print(f"Final state: {results_trans[-1, 0:2]}")
        final_idx = Adjusting_each_body_clock_ini - back_time
        if final_idx-0.1 < 0.0:
            final_idx += 24.0 # 0時を超える場合は24時間を加算
        
        # final_angleが0になる時間を特定すれば、体内時計の位相角度を算出できる。
        for i in range(0, 960):
            final_angle = np.arctan2(results_trans[-i, 1], results_trans[-i, 0]) # 結果の行列から体内時計の位相角度を算出
            if (final_angle < 0.0):
                final_angle += 2 * np.pi
            if (final_idx-0.1) < (final_angle * 24.0 / (2.0 * np.pi)) < (final_idx): # 0.1h以内に0時になるなら
                final_state = results_trans[-i, :]
                final_i = i # 最終状態のインデックスを保存
                break # ループを抜ける

        for j in range(final_i, final_i + 960):
            target_angle = np.arctan2(results_trans[-j, 1], results_trans[-j, 0])
            if (target_angle < 0.0):
                target_angle += 2 * np.pi
            if (final_idx+8-0.05) < (target_angle * 24.0 / (2.0 * np.pi)) < (final_idx+8 + 0.05): # 0.1h以内に0時になるなら
                threshold[0] = results_trans[-j-7, :]
                threshold[1] = results_trans[-j+7, :] # 目標状態の前後7ステップを閾値として保存
                target_state = results_trans[-j, :]
                break # ループを抜ける
        print(f"Final angle(deg): {round(np.rad2deg(final_angle),1)}°, Target angle(deg): {round(np.rad2deg(target_angle),1)}°")
        print(f"Final_angle(rad): {round(final_angle, 2)}/2π, Target angle(rad): {round(target_angle, 2)}/2π")    
        print(f"Final_angle(h)  : {round(final_angle * 24.0 / (2.0 * np.pi), 1)}h, Target angle(backtime時であるべき)(h): {round(target_angle * 24.0 / (2.0 * np.pi), 1)}h")

        """
        # VDPが時計回りに動作するのに対し、反時計回りに動作するため、マイナス1倍します。
        #final_angle = 1.0 * atan2(final_state[1], final_state[0]) # 結果の行列から体内時計の位相角度を算出
        target_angle= 1.0 * atan2(target_state[1], target_state[0]) # 目標状態の位相角度を算出

        if (final_angle < 0.0):
            final_angle += 2 * np.pi # 位相角度が負の場合、2πを加えて正の値に変換
        if (target_angle < 0.0):
            target_angle += 2 * np.pi
        

        print(f"Final angle(deg): {round(np.rad2deg(final_angle),1)}°, Target angle(deg): {round(np.rad2deg(target_angle),1)}°")
        print(f"Final_angle(rad): {round(final_angle, 2)}/2π, Target angle(rad): {round(target_angle, 2)}/2π")
        final_angle = final_angle * 24.0 / (2.0 * np.pi) # 位相角度を24時間制に変換
        target_angle = target_angle * 24.0 / (2.0 * np.pi) # 目標位相角度を24時間制に変換
        print(f"Final_angle(0時であるべき)(h): {round(final_angle, 1)}h, Target angle(backtime時であるべき)(h): {round(target_angle, 1)}h")
        # 結果を返す
        """
        return (final_state, target_state, threshold)# [ini,opt,threshold]



    #後期追加部分ここから
    def update_light_schedule_from_tj(self, t_j, intensity_y, t0_idx, tf_idx, optimize_days=None, office_time=None, lunch_time=None):
        """
        t_j(ON/OFF時刻)に基づき light_schedule を更新する。
        - optimize_days/office_time/lunch_time が与えられれば、オフィス時間(昼休み除外)だけを最適化対象にする。
        office_time: (start_hour, end_hour)
        lunch_time: (start_hour, end_hour) or None
        """
        if office_time is None or optimize_days is None:
            # デフォルト: 期間全体を0にし、t_jのON→OFFで上書き
            self.light_schedule[t0_idx:tf_idx] = 0.0
        else:
            office_start_hour, office_end_hour = office_time
            lunch_start_hour, lunch_end_hour = lunch_time if lunch_time is not None else (None, None)
            for day in range(optimize_days):
                day_start_idx = t0_idx + int(day * 24 / self.dt)
                office_start_idx = day_start_idx + int(office_start_hour / self.dt)
                office_end_idx = day_start_idx + int(office_end_hour / self.dt)
                if lunch_start_hour is not None and lunch_end_hour is not None:
                    lunch_start_idx = day_start_idx + int(lunch_start_hour / self.dt)
                    lunch_end_idx = day_start_idx + int(lunch_end_hour / self.dt)
                    self.light_schedule[office_start_idx:lunch_start_idx] = 0.0
                    self.light_schedule[lunch_end_idx:office_end_idx] = 0.0
                else:
                    self.light_schedule[office_start_idx:office_end_idx] = 0.0
        # t_jのON→OFF区間を intensity_y で上書き
        for i in range(0, len(t_j), 2):
            start_idx = t_j[i]
            end_idx = t_j[i+1] if i + 1 < len(t_j) else tf_idx
            if start_idx < end_idx:
                self.light_schedule[start_idx:end_idx] = intensity_y
    
    def recalculate_dynamics(self, t0_idx, tf_idx):# t_j, intensity_y, tf_idxを除いた
        """t_jに基づいてスケジュールを更新し、再積分する"""
        # self.lightscheduleはstep5で直したので、この関数は不要
        # self.update_light_schedule_from_tj(t_j, intensity_y, t0_idx, tf_idx)
        
        initial_state = self.results[t0_idx] # 最適化開始時の体内時計の位相を指定
        #   sp.integrate.solve_ivp(解く微分方程式,　積分を行う時間範囲,　状態変数の初期条件を示す配列,　解を評価する特定の時間点, 使用する積分アルゴリズム)
        r = sp.integrate.solve_ivp( # 体内時計を再計算
            self.derv,
            (self.ts[t0_idx], self.ts[tf_idx]),
            initial_state,
            t_eval=self.ts[t0_idx:tf_idx+1],
            method='Radau'
        )
        if r.success:
            # 結果を元の配列にマージ
            self.results[t0_idx:tf_idx+1, :] = np.transpose(r.y)
        else:
            print("Warning: Recalculation failed.")
    #後期追加部分ここまで


    """     ---効率化用コードここから---     """
# ...existing code...

    def integrateModel_rk4(self):
        """全期間を固定刻みRK4（numba）で積分して self.results を上書き"""
        # 配列を連続にしておく
        self.results = np.ascontiguousarray(self.results)
        self.light_schedule = np.ascontiguousarray(self.light_schedule)
        # 呼び出し：結果[0] が初期状態であることを仮定
        integrate_forward_numba(self.results, self.light_schedule, 0, self.results.shape[0]-1, self.dt,
                                self.mu, self.G, self.alpha_0, self.delta, self.p, self.I0, self.kparam, self.taux)

    def recalculate_dynamics_rk4(self, t0_idx, tf_idx):
        """t0_idx から tf_idx を RK4(numba) で再積分"""
        # 配列連続化
        self.results = np.ascontiguousarray(self.results)
        self.light_schedule = np.ascontiguousarray(self.light_schedule)
        # 初期値を確実にセット（既存の self.results[t0_idx] を使う）
        # integrate_forward_numba は results[t0_idx] を初期値として用いる
        integrate_forward_numba(self.results, self.light_schedule, t0_idx, tf_idx, self.dt,
                                self.mu, self.G, self.alpha_0, self.delta, self.p, self.I0, self.kparam, self.taux)

    def compare_radau_vs_rk4(self, t0_idx, tf_idx, verbose=True):
        """
        指定区間を Radau 再積分と RK4(numba) 再積分で比較し、
        最大/平均絶対差を返す。self.results は元に戻す。
        """
        backup_results = self.results.copy()

        # Radau（既存実装）
        self.recalculate_dynamics(t0_idx, tf_idx)
        radau_seg = self.results[t0_idx:tf_idx+1].copy()

        # 復元して RK4 実行
        self.results = backup_results.copy()
        self.recalculate_dynamics_rk4(t0_idx, tf_idx)
        rk4_seg = self.results[t0_idx:tf_idx+1].copy()

        # 誤差計算
        diff = np.abs(radau_seg - rk4_seg)
        max_err = diff.max()
        mean_err = diff.mean()

        if verbose:
            print(f"compare_radau_vs_rk4: max_err={max_err:.3e}, mean_err={mean_err:.3e}")

        # 復元
        self.results = backup_results
        return max_err, mean_err
    
    def compare_adjoint_methods(self, opt_target, tf_idx, t0_idx, verbose=True):
        """
        随伴方程式の計算について、Radau法とnumba版(後退オイラー)の結果を比較する。
        self.lambda_t は元に戻す。
        """
        # 現在の lambda_t をバックアップ
        backup_lambda_t = self.lambda_t.copy()

        # --- Radau法で計算 ---
        self.calculate_adjoint_dynamics(opt_target, tf_idx)
        lambda_radau = self.lambda_t[t0_idx:tf_idx+1].copy()

        # --- numba版で計算 ---
        self.calculate_adjoint_dynamics_numba(opt_target, tf_idx)
        lambda_numba = self.lambda_t[t0_idx:tf_idx+1].copy()

        # --- 誤差を計算 ---
        diff = np.abs(lambda_radau - lambda_numba)
        max_err = diff.max()
        mean_err = diff.mean()

        if verbose:
            print(f"compare_adjoint_methods: max_err={max_err:.3e}, mean_err={mean_err:.3e}")

        # バックアップから復元
        self.lambda_t = backup_lambda_t
        return max_err, mean_err
    
    # ...existing code...
        # --- ここから新しいメソッドを追加 ---
    def calculate_adjoint_dynamics_numba(self, opt_target, tf_idx, t0_idx):
        """
        numbaベースの固定刻み法で随伴方程式を高速に逆積分する。
        """
        # 終端λの計算
        x_tf = self.results[tf_idx]
        error = x_tf[0:2] - opt_target[0:2]
        cost_J = np.sqrt(np.sum(error**2))
        
        lambda_tf = np.zeros(3)
        if cost_J > 1e-6:
            lambda_tf[0] = error[0] / cost_J
            lambda_tf[1] = error[1] / cost_J
        
        # ヤコビ行列を全時刻で事前計算
        jacobians = _compute_jacobians_njit(self.results, self.light_schedule, self.dt,
                                            self.mu, self.G, self.alpha_0, self.delta,
                                            self.p, self.I0, self.kparam, self.taux)
        
        # 逆積分
        self.lambda_t.fill(0)
        self.lambda_t[tf_idx] = lambda_tf
        _integrate_adjoint_backward_euler_njit(self.lambda_t, jacobians, t0_idx, tf_idx, self.dt)
    # --- 新しいメソッドここまで ---

    """     ---効率化用コードここまで---     """


    #step2ここから
    def recalculate_dynamics_pre(self, t_j, intensity_y, t0_idx, tf_idx):
        """t_jに基づいてスケジュールを更新し、再積分する"""
        self.update_light_schedule_from_tj(t_j, intensity_y, t0_idx, tf_idx)
        
        initial_state = self.results[t0_idx] # 最適化開始時の体内時計の位相を指定
        #   sp.integrate.solve_ivp(解く微分方程式,　積分を行う時間範囲,　状態変数の初期条件を示す配列,　解を評価する特定の時間点, 使用する積分アルゴリズム)
        r = sp.integrate.solve_ivp( # 体内時計を再計算
            self.derv,
            (self.ts[t0_idx], self.ts[tf_idx]),
            initial_state,
            t_eval=self.ts[t0_idx:tf_idx+1],
            method='Radau'
        )
        if r.success:
            # 結果を元の配列にマージ
            self.results[t0_idx:tf_idx+1, :] = np.transpose(r.y)
        else:
            print("Warning: Recalculation failed.")
    # step2ここまで(derv関数含む)
    

    # step3ここから
    def _get_jacobian(self, t, y):
        x, xc, n = y
        idx = int(round(t / self.dt)); idx = min(idx, len(self.light_schedule) - 1)
        current_light = self.light_schedule[idx]
        current_alpha = self.alpha0(current_light)
        
        B_hat = self.G * (1.0 - n) * current_alpha
        B = B_hat * (1 - 0.4 * x) * (1 - 0.4 * xc)
        
        # 偏微分 dB/dx, dB/dxc, dB/dn
        dBdx = B_hat * (-0.4) * (1 - 0.4 * xc)
        dBdxc = B_hat * (1 - 0.4 * x) * (-0.4)
        dBdn = self.G * (-1.0) * current_alpha * (1 - 0.4 * x) * (1 - 0.4 * xc)
        
        dfdx = np.zeros((3, 3))
        
        # --- 行 0 ---
        dfdx[0, 0] = (np.pi / 12.0) * dBdx
        dfdx[0, 1] = (np.pi / 12.0) * (1.0 + dBdxc)
        dfdx[0, 2] = (np.pi / 12.0) * dBdn
        
        # --- 行 1 (★最終修正★) ---
        C1 = pow(24.0 / (0.99669 * self.taux), 2.0)
        
        
        # 正しい偏微分を適用
        # ∂/∂x [ -C1*x - k*x*B ]
        dfdx[1, 0] = -(np.pi / 12.0) * (C1 + (self.kparam * x * dBdx) + (self.kparam * B))
        
        # ∂/∂xc [ μ*(...) - k*x*B ]
        dfdx[1, 1] = (np.pi / 12.0) * (self.mu * (1.0 - 4.0 * xc**2) - self.kparam * x * dBdxc)

        # ∂/∂n [ -k*x*B ]
        dfdx[1, 2] = -(np.pi / 12.0) * (self.kparam * x * dBdn)
        
        # --- 行 2 ---
        dfdx[2, 0] = 0.0
        dfdx[2, 1] = 0.0
        dfdx[2, 2] = -60.0 * (current_alpha + self.delta)
        
        return dfdx
    
    # step3
    def derv_adjoint(self, t, lambda_val):
        """随伴方程式 dλ/dt = - (∂f/∂x)^T λ"""
        # 順方向積分の結果(self.results)を補間して、時刻tでの状態yを取得
        y_t = self.interp_results(t)
        
        # 時刻tでのヤコビアンを取得
        jacobian_t = self._get_jacobian(t, y_t)
        
        # 随伴方程式の計算
        dlambda_dt = -np.dot(jacobian_t.T, lambda_val)
        return dlambda_dt
    
    # step3
    def calculate_adjoint_dynamics(self, opt_target, tf_idx, t0_idx):
        """
        最終時刻tf_idxから逆方向に随伴方程式を解く (Step 3)
        """
        # 逆方向積分のため、順方向の結果を補間する関数を用意
        self.interp_results = interp1d(self.ts, self.results, axis=0, bounds_error=False, fill_value="extrapolate")
        
        # Step 3.1: 最終時刻tfでのλの初期値を計算
        # λ(tf) = ∂J/∂x(tf)
        # J = sqrt( (x-xt)^2 + (xc-xct)^2 )
        # ∂J/∂x = (x-xt) / J
        # ∂J/∂xc = (xc-xct) / J
        x_tf = self.results[tf_idx]
        x_target = opt_target[0:2] # 目標はxとxcのみ
        
        error = x_tf[0:2] - x_target
        cost_J = np.sqrt(np.sum(error**2))
        
        lambda_tf = np.zeros(3)
        if cost_J > 1e-6: # ゼロ除算を避ける
            lambda_tf[0] = error[0] / cost_J
            lambda_tf[1] = error[1] / cost_J
        # lambda_n(tf) は 0
        
        # Step 3.2: 逆方向積分
        t_start_opt = self.ts[t0_idx]
        t_final_opt = self.ts[tf_idx]

        r_adjoint = sp.integrate.solve_ivp(
            self.derv_adjoint,
            (t_final_opt, t_start_opt), # ★★★ 逆向きの時間範囲 ★★★
            lambda_tf,# 初期値
            t_eval=np.flip(self.ts[t0_idx:tf_idx+1]), # ★★★ 評価点も逆順 ★★★
            method='Radau'
        )
        
        if r_adjoint.success:
            # 結果を正しい順序に戻して保存
            self.lambda_t[t0_idx:tf_idx+1] = np.flip(np.transpose(r_adjoint.y), axis=0)
        else:
            print(f"Warning: Adjoint integration failed for taux={self.taux}")
        # step3ここまで



    # step4ここから
    def _get_df_dI(self, t, y, homelux=150.0):
        """
        微分方程式 f を、現在の光強度 I で偏微分したベクトル (∂f/∂I) を計算する。
        """
        x, xc, n = y
        idx = int(round(t / self.dt)); idx = min(idx, len(self.light_schedule) - 1)
        current_intensity = self.light_schedule[idx]

        if current_intensity <= 1e-6:
            current_intensity = homelux

        d_alpha_d_I = self.alpha_0 * self.p * pow(current_intensity, self.p - 1) / pow(self.I0, self.p)
        
        #B = self.G * (1.0 - n) * self.alpha0(current_intensity) * (1 - 0.4 * x) / (1 - 0.4 * xc) # αから逆算　なんかいらない処理してた
        d_B_d_I = self.G * (1.0 - n) * (1 - 0.4 * x) * (1 - 0.4 * xc) * d_alpha_d_I # Bはαに比例するため

        df_dI = np.zeros(3)
        df_dI[0] = (np.pi / 12.0) * d_B_d_I
        df_dI[1] = -(np.pi / 12.0) * self.kparam * x * d_B_d_I
        df_dI[2] = 60.0 * d_alpha_d_I * (1.0 - n)
        
        return df_dI


    def calculate_sensitivity_for_intensity(self, start_idx, end_idx):
        """
        指定された明期（start_idx から end_idx）の明るさに対するコストJの感度を計算する。
        
        Args:
            start_idx (int): 明期の開始インデックス。
            end_idx (int): 明期の終了インデックス。
        
        Returns:
            float: 計算された感度 (∂J/∂I)。
        """
        sensitivity = 0.0
        # 指定された明期の期間でループして積分
        for i in range(start_idx, end_idx):
            if i >= len(self.ts):
                break
            t = self.ts[i]
            y = self.results[i]
            lambda_val = self.lambda_t[i]
            df_dI = self._get_df_dI(t, y)
            sensitivity += np.dot(lambda_val, df_dI) # 1時間分6分毎の感度10個を足し合わせている
        return  -(sensitivity * self.dt)


    def calculate_sensitivity_for_intensity_tf(self, tf_idx):
        """
        指定された明期（start_idx から end_idx）の明るさに対するコストJの感度を計算する。
        
        Args:
            start_idx (int): 明期の開始インデックス。
            end_idx (int): 明期の終了インデックス。
        
        Returns:
            float: 計算された感度 (∂J/∂I)。
        """
        sensitivity = 0.0
        # 指定された明期の期間でループして積分
        t = self.ts[tf_idx]
        y = self.results[tf_idx]
        lambda_val_tf = self.lambda_t[tf_idx]
        df_dI_tf = self._get_df_dI(t, y)
        sensitivity = np.dot(lambda_val_tf, df_dI_tf) # 1時間分6分毎の感度10個を足し合わせている
        return (sensitivity * self.dt)
    


    def calculate_sensitivity_for_intensity_t_j(self, t_j):
        """
        指定された明期（start_idx から end_idx）の明るさに対するコストJの感度を計算する。
        
        Args:
            start_idx (int): 明期の開始インデックス。
            end_idx (int): 明期の終了インデックス。
        
        Returns:
            float: 計算された感度 (∂J/∂I)。
        """
        sensitivity = 0.0
        # 指定された明期の期間でループして積分
        t = self.ts[t_j]
        y = self.results[t_j]
        lambda_val = self.lambda_t[t_j]
        df_dI = self._get_df_dI(t, y)
        sensitivity = np.dot(lambda_val, df_dI) 
        return  -sensitivity
    # step4ここまで

    def finalcalc(self, t0_idx,tf_idx):# t_j, intensity_y, tf_idxを除いた
        """t_jに基づいてスケジュールを更新し、再積分する"""
        # self.lightscheduleはstep5で直したので、この関数は不要
        # self.update_light_schedule_from_tj(t_j, intensity_y, t0_idx, tf_idx)

        initial_state = self.results[t0_idx] # 最適化開始時の体内時計の位相を指定
        #   sp.integrate.solve_ivp(解く微分方程式,　積分を行う時間範囲,　状態変数の初期条件を示す配列,　解を評価する特定の時間点, 使用する積分アルゴリズム)
        r = sp.integrate.solve_ivp( # 体内時計を再計算
            self.derv,
            (self.ts[t0_idx], self.ts[-1]),
            initial_state,
            t_eval=self.ts[t0_idx:],
            method='Radau'
        )
        if r.success:
            # 結果を元の配列にマージ
            self.results[t0_idx:,:] = np.transpose(r.y)
        else:
            print("Warning: Recalculation failed.")

    def calculate_dtj(self, j, t_j):
        """
        スイッチング時刻t_jの最適な移動量を計算する (Step 4)
        j: スイッチング時刻のインデックス
        t_j: スイッチング時刻の配列 (全スイッチ)
        """
        # スイッチング時刻のインデックスを取得
        t_j_alta = t_j[j]

        # --- H(t_j^-) の計算 (スイッチ直前) ---
        # スイッチ直前の状態と感度を取得
        x_before = self.results[t_j_alta - 1]
        lambda_before = self.lambda_t[t_j_alta - 1]
        # スイッチ直前の光強度 (このメソッドが呼び出される前にlight_scheduleは更新済み)
        # なので、単にその時刻の光強度を使えば良い
        # f(x(t-), u(t-)) を計算
        f_before = self.derv(self.ts[t_j_alta - 1], x_before)
        # H(t-) = λ(t-)^T * f(x(t-), u(t-))
        H_before = np.dot(lambda_before, f_before)

        # --- H(t_j^+) の計算 (スイチ直後) ---
        # スイッチ直後の状態と感度は、直前のものと同じと近似できる
        x_after = self.results[t_j_alta]
        lambda_after = self.lambda_t[t_j_alta]
        # f(x(t+), u(t+)) を計算
        # ★重要★: dervは内部でself.light_schedule[t_j_alta]を見るので、
        # スイッチ後の光強度での計算が自動的に行われる
        f_after = self.derv(self.ts[t_j_alta], x_after)
        # H(t+) = λ(t+)^T * f(x(t+), u(t+))
        H_after = np.dot(lambda_after, f_after)

        # --- スイッチング感度と移動量の計算 ---
        # ∂J/∂t_j = H(t_j-) - H(t_j+)
        switching_sensitivity = H_before - H_after
        
        # dt_j ∝ - (∂J/∂t_j)
        # 論文ではステップサイズtsをかけているが、それはメインループで制御する。
        # ここでは感度そのものを返す（あるいは-1をかけたもの）。
        # 符号を反転させて、コストが下がる方向を返す。
        return -switching_sensitivity

    def getTS(self):
        """
        システムの時系列データフレームを返します。

        getTS()

        このVDPモデルの振幅は、位相空間の(x, x_c)座標から原点までのユークリッド距離として定義されます。

        アンラップされた位相推定は、位相平面座標の-1*arctanとして取得されます。
        これは、Hannayモデルと比較できるように変換されています。

        Time、Light_Level (lux)、Phase (radians)、R (振幅)、
        n (光活性化変数) を列とするpandasデータフレームを返します。
        """
        
        light_ts = self.light_schedule  # list()list型に型変換する関数 #map()各要素に対して指定した関数を適用し、結果を返す便利な組み込み関数
        # シミュレーション期間の配列tsにLightFunReg(つまりRegularLightSimple関数)の処理を適用して、その時刻における光のレベル（強度）を計算
        "ここで光の強度を決めている。なので、ここの処理を「最適化」や「家庭スケジュール」に置き換える"
        # 振幅を各座標の2乗の平方根として定義します
        Amplitude = np.sqrt(self.results[:, 0] ** 2 + self.results[:, 1] ** 2) #振幅を計算するために、位相空間の(x, x_c)座標から原点までのユークリッド距離を計算

        # 位相をラジアンで抽出する必要があります
        wrappedPhase = -1.0 * \
            np.arctan2(self.results[:, 1], self.results[:, 0]) # さっきと似たような処理で位相をラジアンにしている？

        # 0から2piの間にします
        for i in range(len(wrappedPhase)): # さっきと似たような処理で位相を正にしている？
            if wrappedPhase[i] < 0.0:
                wrappedPhase[i] += 2 * np.pi

        Phase = np.unwrap(wrappedPhase, discont=0.0) # np.unwrap()は、位相の不連続性を解消する。0～2πの範囲に絞ることでで失われる連続性を取り戻すために、位相をアンラップする関数です。
        # Phaseに連続的な位相を格納
        # データフレームを作成します
        ts = pd.DataFrame({ 'Time': self.ts,            # シミュレーション期間の時間を表す列
                            'Light_Level': light_ts,    # シミュレーション期間の時間ごとの光レベルを表す列
                            'Phase': Phase,             # 体内時計の連続的な位相
                            'R': Amplitude,             # 体内時計の振幅
                            'x': self.results[:, 0],    # 位相空間のx座標
                            'xc': self.results[:, 1],   # 位相空間のx_c座標
                            'n': self.results[:, 2],})  # 光活性化変数n
                            # 微分方程式の解　nは光活性化変数
        return (ts)


def guessICDataVDP(LightFunc, time_zero, length=50):
    """個人の光スケジュールを使用してモデルの初期条件を推測します。"""

    # Forgerモデルのインスタンスを作成（デフォルト設定）
    a = vdp_forger99_model(24.2, int(length) * 24.0, 0.1)
    # 光スケジュールを関数から生成
    diff_light = np.array([LightFunc(t) for t in a.ts])
    a.set_initial_diff_light_schedule(diff_light)
    # 初期位相を大まかに推測します
    init = np.array([1.0, 1.0, 0.0])

    a.integrateModel(int(length) * 24.0, initial=init)
    init = a.results[-1, :]
    a.integrateModel(48.0, initial=init)

    limit_cycle = a.results
    def timeDay(x): return fmod(x, 48.0)
    lc_ts = np.array(list(map(timeDay, a.ts)))

    idx = np.searchsorted(lc_ts, time_zero) - 1
    initial = limit_cycle[idx, :]
    # print time_zero, initial
    return (initial)

if __name__ == '__main__':
    # モジュール単体実行時の簡易チェック（未使用の vdp_model 参照を削除）
    pass