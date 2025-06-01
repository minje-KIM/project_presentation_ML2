import torch
import torch.nn as nn
import torch.nn.functional as F
from rita_observation_model import DeepVIO
from model_for_classification import IMUClassifier

def is_stationary(imu_acc, imu_ang, threshold=0.01):
    """
    IMU 가속도/각속도의 변화량을 기준으로 객체가 정지 상태인지 판별
    """
    
    acc_variation = torch.mean(torch.abs(imu_acc[:, -1, :] - imu_acc[:, 0, :]), dim=-1) 
    ang_variation = torch.mean(torch.abs(imu_ang[:, -1, :] - imu_ang[:, 0, :]), dim=-1) 

    stationary = (acc_variation < threshold) & (ang_variation < threshold)
    return stationary

class TransitionModel(nn.Module):
    """
    IMU와 이전 상태를 기반으로 다음 상태를 예측하는 Kalman Filter의 전이 모델
    좌회전 / 직진 / 우회전의 세 가지 동작에 대해 별도의 네트워크로 분기
    """
    
    def __init__(self):
        super(TransitionModel, self).__init__()
        
        self.layer_init = nn.Sequential(
            nn.Linear(6, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
        
        # imu 데이터를 split한 LSTM 인코더
        self.imu_encoder  = nn.LSTM(input_size=6, hidden_size=256, num_layers=2, bidirectional=False, batch_first=True)
        self.ang_encoder  = nn.LSTM(input_size=3, hidden_size=256, num_layers=2, bidirectional=False, batch_first=True)

        # 좌회전, 직진, 우회전 각각에 대해 회전(R), 이동(T), A, Q 구성
        self.layer_t_left = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.layer_r_left = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )        
       
        self.layer_t_straight = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.layer_r_straight = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )    
                    
        self.layer_t_right = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.layer_r_right = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )      

        self.layer_Q_left = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )
        
        self.layer_A_left = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 12)
        )                

        self.layer_Q_straight = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )
        
        self.layer_A_straight = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 12)
        )      

        self.layer_Q_right = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )
        
        self.layer_A_right = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 12)
        )      

        # Xavier initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
                
    def forward(self, xt_minus_1, imu_acc, imu_ang, decision_weights):
        """
        이전 상태와 IMU 입력값 및 movement 분류 확률을 이용해
        다음 상태 예측값, 전이 행렬 A, 노이즈 공분산 Q를 출력
        """
        
        imu = torch.cat((imu_acc, imu_ang), dim=-1) # (16, 11, 6)
        xt = self.layer_init(xt_minus_1)     # (16, 6) -> (16, 256)  
              
        imu_feat, _ = self.imu_encoder(imu)        # (16, 11, 256)
        imu_feat = imu_feat[:,-1,:]                 # (16, 256)

        ang_feat, _ = self.ang_encoder(imu_ang)    # (16, 11, 256)
        ang_feat = ang_feat[:,-1,:]                 # (16, 256)

        feat_t = torch.cat((xt, imu_feat), dim=1)   # (16, 512)
        feat_r = torch.cat((xt, ang_feat), dim=1)   # (16, 512)

        tra_left = self.layer_t_left(feat_t)  # (16, 3)
        rot_left = self.layer_r_left(feat_r)  # (16, 3)
        
        tra_straight = self.layer_t_straight(feat_t)  # (16, 3)
        rot_straight = self.layer_r_straight(feat_r)  # (16, 3)        

        tra_right = self.layer_t_right(feat_t)  # (16, 3)
        rot_right = self.layer_r_right(feat_r)  # (16, 3)

        xt_prime_left = torch.cat((rot_left, tra_left), dim=1)  # (16, 6)
        xt_prime_straight = torch.cat((rot_straight, tra_straight), dim=1)  # (16, 6)
        xt_prime_right = torch.cat((rot_right, tra_right), dim=1)  # (16, 6)

        A_left = self.layer_A_left(feat_t)     # (16, 12)
        Q_left = self.layer_Q_left(feat_t)     # (16, 9)    
            
        A_straight = self.layer_A_straight(feat_t)     # (16, 12)
        Q_straight = self.layer_Q_straight(feat_t)     # (16, 9)   
         
        A_right = self.layer_A_right(feat_t)     # (16, 12)
        Q_right = self.layer_Q_right(feat_t)     # (16, 9)            
        
        A_left, Q_left = self.compute_A_Q(A_left, Q_left)
        A_straight, Q_straight = self.compute_A_Q(A_straight, Q_straight)
        A_right, Q_right = self.compute_A_Q(A_right, Q_right)        
        
        xt_prime = decision_weights[:, 0].unsqueeze(1) * xt_prime_left + \
                   decision_weights[:, 1].unsqueeze(1) * xt_prime_straight + \
                   decision_weights[:, 2].unsqueeze(1) * xt_prime_right

        A = decision_weights[:, 0].unsqueeze(1).unsqueeze(1) * A_left + \
            decision_weights[:, 1].unsqueeze(1).unsqueeze(1) * A_straight + \
            decision_weights[:, 2].unsqueeze(1).unsqueeze(1) * A_right

        Q = decision_weights[:, 0].unsqueeze(1).unsqueeze(1) * Q_left + \
            decision_weights[:, 1].unsqueeze(1).unsqueeze(1) * Q_straight + \
            decision_weights[:, 2].unsqueeze(1).unsqueeze(1) * Q_right
        
        return xt_prime, A, Q
        
    def compute_A_Q(self, At, Qt):

        # A matrix computation
        A = torch.zeros(At.size(0), 6, 6).to(At.device)
        A[:, 0, 0] = At[:, 0]
        A[:, 1, 1] = At[:, 1]
        A[:, 2, 2] = At[:, 2]
        A[:, 3, 3] = At[:, 3]
        A[:, 4, 4] = At[:, 4]
        A[:, 5, 5] = At[:, 5]
        A[:, 1, 3] = At[:, 6]
        A[:, 1, 5] = At[:, 7]
        A[:, 3, 1] = At[:, 8]
        A[:, 3, 5] = At[:, 9]
        A[:, 5, 1] = At[:, 10]
        A[:, 5, 3] = At[:, 11]

        # Q matrix computation
        Q = torch.zeros(Qt.size(0), 6, 6).to(Qt.device)
        Q[:, 0, 0] = Qt[:, 0] ** 2  # sigma roll (X)
        Q[:, 1, 1] = Qt[:, 1] ** 2  # sigma yaw (Y)
        Q[:, 2, 2] = Qt[:, 2] ** 2  # sigma pitch (Z)
        Q[:, 3, 3] = Qt[:, 3] ** 2  # sigma x
        Q[:, 4, 4] = Qt[:, 4] ** 2  # sigma y
        Q[:, 5, 5] = Qt[:, 5] ** 2  # sigma z

        Q[:, 3, 5] = Qt[:, 6] * Qt[:, 3] * Qt[:, 5]  # raw1 * sigma x * sigma z
        Q[:, 5, 3] = Qt[:, 6] * Qt[:, 3] * Qt[:, 5]  # raw1 * sigma x * sigma z

        Q[:, 3, 1] = Qt[:, 7] * Qt[:, 3] * Qt[:, 1]  # raw2 * sigma x * sigma Y
        Q[:, 1, 3] = Qt[:, 7] * Qt[:, 3] * Qt[:, 1]  # raw2 * sigma x * sigma Y

        Q[:, 1, 5] = Qt[:, 8] * Qt[:, 5] * Qt[:, 1]  # raw3 * sigma z * sigma Y
        Q[:, 5, 1] = Qt[:, 8] * Qt[:, 5] * Qt[:, 1]  # raw3 * sigma z * sigma Y

        return A, Q    

class LSTMPrior(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMPrior, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 36)

    def forward(self, K_prime, hidden):
        out, hidden = self.lstm(K_prime, hidden)
        out = self.linear(out)
        return out, hidden

class Kalman_VIO(nn.Module):
    """
    Kalman 구조를 따르는 VIO 통합 네트워크
    Observation 모델 + Transition 모델 + Kalman 업데이트 로직을 통합
    """
    
    def __init__(self, opt):
        super(Kalman_VIO, self).__init__()
        self.opt = opt
        self.observation_model = DeepVIO(self.opt)
        self.lstm_prior = LSTMPrior(36, 128, 2)
        self.transition_model = TransitionModel()
        self.classifier = IMUClassifier()
        
    def forward(self, img, imu, all_imu_samples, is_first=True, hc=None, temp=5, selection='gumbel-softmax', p=0.5):
        """
        Kalman 기반 VIO 처리의 메인 함수
        이미지 + IMU → 관측값 zt, movement 분류 → Transition 모델 예측값 → 칼만 업데이트 순으로 진행
        """
        
        # Observation 모델을 통해 zt와 R, 분류 결과 추출
        zt, R, decisions, probs, hc, fi = self.observation_model(img, imu, is_first=True, hc=None, temp=temp, selection=selection, p=p)
        
        # zt (16, 10, 6) / H,R (16, 10, 6, 6)        
        
        batch_size = img.size(0)
        seq_len = img.size(1) -1
                
        # movement classifier 입력을 위해 과거 IMU 시퀀스를 슬라이싱
        splitted_imu = []
        for i in range(10):
            start_idx = i * 10
            end_idx = start_idx + ((self.opt.past_len * 10) + 1)
            sliced_data = all_imu_samples[:, start_idx:end_idx, :]  
            splitted_imu.append(sliced_data)


        splitted_imu = torch.stack(splitted_imu, dim=1)
        
        # 각 time step별 movement classification 수행
        output_list = []
        for i in range(10):
            input_slice = splitted_imu[:, i, :, :]
            output_slice  = self.classifier(input_slice)
            output_list.append(output_slice)
        classified_weight = torch.stack(output_list, dim=1)     # (16, 10, 101, 6)


        xt = torch.zeros(batch_size, 6).to(zt.device)        # (16, 6)
        imu = torch.cat([imu[:, i * 10:i * 10 + 11, :].unsqueeze(1) for i in range(seq_len)], dim=1)
        imu_acc, imu_ang = torch.split(imu, 3, dim=-1)
        
        # 초기 공분산 및 LSTM 히든 상태 설정
        P = torch.eye(6).unsqueeze(0).repeat(batch_size, 1, 1).to(xt.device) # (16, 6, 6)
        lstm_hidden = (torch.zeros(2, batch_size, 128).to(xt.device), torch.zeros(2, batch_size, 128).to(xt.device))

        xt_history = []
        trans_history = []
        P_history = []
        Q_history = []
        A_history = []
        K_history = []
        Axt_history = []

        for t in range(10):
            stationary = is_stationary(imu_acc[:, t, :, :], imu_ang[:, t, :, :])

            # 전이 모델을 통해 다음 상태 및 행렬 A, Q 예측
            xt_prime, A, Q = self.transition_model(xt, imu_acc[:,t,:,:], imu_ang[:,t,:,:], classified_weight[:,t,:])
            trans_history.append(xt_prime.unsqueeze(1))
            Q_history.append(Q.unsqueeze(1))
            A_history.append(A.unsqueeze(1))
            
            axt = torch.bmm(A, xt.unsqueeze(-1)).squeeze(2)
            
            # Kalman 업데이트 수행
            xt, P, K , lstm_hidden = self.kalman_update(xt_prime, zt[:,t,:], A, Q, R[:, t, :, :], P, lstm_hidden, stationary)
            xt_history.append(xt.unsqueeze(1))
            P_history.append(P.unsqueeze(1))
            K_history.append(K.unsqueeze(1))
        
            Axt_history.append(axt.unsqueeze(1))

        trans_history = torch.cat(trans_history, dim=1)
        xt_history = torch.cat(xt_history, dim=1)
        P_history = torch.cat(P_history, dim=1)
        Q_history = torch.cat(Q_history, dim=1)
        A_history = torch.cat(A_history, dim=1)        
        K_history = torch.cat(K_history, dim=1)        
        Axt_history = torch.cat(Axt_history, dim=1)

        return trans_history, zt, xt_history, Axt_history, R, Q_history, P_history, A_history, K_history, decisions, probs, hc
        

    def kalman_update(self, xt_prime, zt, A, Q, R, P, lstm_hidden, stationary):
        """
        Kalman 필터 업데이트 단계
        - 정지 상태 시 관측값만 사용
        - 움직이는 경우 Kalman Gain 계산하여 상태 갱신
        """
        
        I = torch.eye(6).to(xt_prime.device)
        
        # 예측 공분산 계산
        P_prime = torch.bmm(A, P)
        P_prime = torch.bmm(P_prime, A.transpose(1,2)) + Q
        
        # 정지 상태이면 Kalman Gain을 단위 행렬로
        if stationary.sum() > 0:
            Kt_prime = torch.eye(6).unsqueeze(0).repeat(P_prime.size(0), 1, 1).to(P_prime.device)
        else:
            # 정지 상태가 아니면 일반적인 Kalman Gain 계산
            Kt_prime = self.predict_kalman_gain(P_prime, R)
        
        # LSTM을 이용해 Kalman Gain 보정
        Kt_prime_res = Kt_prime.view(Kt_prime.size(0), 1, -1)       # (16, 1, 36)
        K, lstm_hidden = self.lstm_prior(Kt_prime_res, lstm_hidden)
        K = K.view(K.size(0), 6, 6)
        
        # 상태 업데이트
        xt_prime = xt_prime.unsqueeze(-1)  # (batch_size, 6, 1)
        zt = zt.unsqueeze(-1)  # (batch_size, 6, 1)

        if stationary.sum() > 0:
            xt = zt.squeeze(-1)
        else:
            xt = torch.bmm(I - K, xt_prime) + torch.bmm(K, zt)  # (batch_size, 6, 1)
            xt = xt.squeeze(-1)  # (batch_size, 6)

        P = torch.bmm((I - K), P_prime)
        P = torch.bmm(P, (I - K).transpose(1, 2)) + torch.bmm(K, torch.bmm(R, K.transpose(1, 2)))
        
        return xt, P, K, lstm_hidden
    
    def predict_kalman_gain(self, P_prime, Rt):
        """
        Kalman Gain 계산: K = P * (P + R)^-1
        """
        
        S = P_prime + Rt
        Kt_prime = torch.bmm(P_prime, torch.inverse(S))
        return Kt_prime   


