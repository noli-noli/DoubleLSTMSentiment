import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from collections import Counter
import os
#可視化
import plotly.graph_objs as go
import time
import numpy as np




"""
モデルアーキテクチャの定義
[input] -> embedding -> LSTM -> Dense -> LSTM -> [output]
"""

class DoubleLSTMSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super(DoubleLSTMSentiment, self).__init__()
        self.Embedding = nn.Embedding(vocab_size, embed_dim)            # 埋め込み層
        self.rnn1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True)    # LSTM層 1層目
        self.dense = nn.Linear(hidden_dim, hidden_dim)                  # Dense層
        self.rnn2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)   # LSTMレイヤー　2層目
        self.fc = nn.Linear(hidden_dim, num_class)                      # Linear層

    def forward(self, text):
        # XXX: tensorの形を確認する
        embedded = self.Embedding(text)                                 # 埋め込み層
        output1, (hidden1, cell1) = self.rnn1(embedded)                 # LSTM 1層目
        dense_output = self.dense(output1)                              # Dense層 (LSTM 層 1層目の出力をDense レイヤーに入力)
        output2, (hidden2, cell2) = self.rnn2(dense_output)             # LSTM 2層目 (Dense レイヤーの出力をLSTM 層 2層目に入力)

        return self.fc(output2[:,-1:,:]).view(1)

class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0, path="checkpoint.pt", trace_func=print):
        """
        Args:
            patience (int): どれだけのエポックの間改善がない場合にトレーニングを停止するかの設定
            verbose (bool): Trueの場合、各エポックでのモデルの保存メッセージを表示
            delta (float): この値よりも小さな改善は無視される
            path (str): モデルの保存先のパス
            trace_func (function): メッセージを表示する関数
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """モデルを保存する"""
        if self.verbose:
            self.trace_func(f"\rValidation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...", end="")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def load_data(file_name):
    """
    データをロードしてリスト化
    """
    print("Loading data...")
    df = pd.read_csv(file_name, encoding="utf-8")
    train_data = df.values.tolist()
    batch_size = len(train_data)
    return train_data, batch_size

def create_vocab(train_data):
    """
    各単語にIDを割り振る
    IDは出現順に割り振っている
    ここの語彙を使ってテストデータも評価する
    #NOTE: ストップワードの除去などの前処理は行っていない
    """
    print("Creating vocabulary...")
    wordFreq = Counter(word for label, text in train_data for word in text.split()) #各単語の出現回数をカウント
    uniqueWords = list(wordFreq.keys())#一意な単語をリスト化
    uniqueWords.insert(0, "<PAD>")# 文章の長さを揃えるためのパディング
    uniqueWords.insert(1, "<UNK>")# 未知語に対応するため特殊なトークンを追加
    vocab = {word: i+2 for i, word in enumerate(uniqueWords)}#単語にIDを割り振る
    return vocab, uniqueWords

def encode_texts(data, vocab):
    """
    作成した語彙を使ってテキストをIDに変換
    testデータでも同じ語彙を使って変換する
    """
    print("Encoding texts...")
    lengths = [len(str(text).split()) for _, text in data]# すべての文の長さを取得
    max_length = max(lengths)  # 最大の文の長さを取得
    #textをIDに変換
    for entry in data:
        words = entry[1].split()
        encoded_words = [vocab[word] if word in vocab else 1 for word in words]
        while len(encoded_words) < max_length:
            encoded_words.append(0)
        entry[1] = encoded_words
    return data
        

def main():
    #params
    VOCAB_SIZE = 100000
    EMBED_DIM = 100
    HIDDEN_DIM = 128
    OUTPUT_DIM = 1
    EPOCH = 10
    LEARNING_RATE = 0.01
    TRAIN_DATA_PATH = "train.csv"
    TEST_DATA_PATH = "test.csv"
   
    """
    GPUの確認
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
    print(f"Device:{device}...")
    
    """
    前処理
    """
    train_data,batch_size = load_data(TRAIN_DATA_PATH)
    vocab, uniqueWords = create_vocab(train_data)
    encoded_train_data = encode_texts(train_data, vocab)
    
    if not os.path.exists("checkpoint.pt"):
        
        """
        モデルのインスタンス化&デバイスの設定
        """
        model = DoubleLSTMSentiment(VOCAB_SIZE,EMBED_DIM,HIDDEN_DIM,OUTPUT_DIM)
        model = model.to(device)
        print("Model created...")
        
        """
        損失関数＆オプティマイザ
        """
        criterion = nn.BCEWithLogitsLoss()  #二値分類なのでBCELossを使用
        #criterion = nn.MSELoss()    #二乗誤差を使用する場合はnn.MSELoss()
        optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
        print("Loss function and optimizer created...")
        
        """
        可視化
        """
        # FIXME: docker上だとなぜかグラフが表示されない
        
        # 初期グラフの設定
        # x_vals = list(range(batch_size+10))
        # y_vals = [0] * batch_size # 初期の損失は0とする
        # trace = go.Scatter(x=x_vals, y=y_vals, mode="lines+markers")
        # layout = go.Layout(title="loss/mini-batch")
        # fig = go.FigureWidget(data=[trace], layout=layout)
        # display(fig)
        # global_step = 0 # ミニバッチごとの損失のログのためのカウンター

        
        """
        学習開始
        """
       
        early_stopping = EarlyStopping(patience=7, verbose=True) # アーリーストッピングのインスタンス化
        
        print("Training...")
        for epoch in range(EPOCH): #学習回数
            for label,text in train_data: #データ数
                inputs = torch.tensor(text).unsqueeze(0).to(device)#入力データをテンソルに変換
                label = torch.tensor(int(label),dtype=torch.float32).unsqueeze(0).to(device)#ラベルをテンソルに変換 #XXX: unsqueeze(0)はなぜ必要なのか
                
                #勾配の初期化 これをしないと前回の勾配が残ってしまう
                #これはミニバッチごとに勾配を初期化している
                optimizer.zero_grad()
                
                
                outputs = model(inputs)
                """
                    上でインスタンス化したモデルに入力、embedding,LSTM,linaerを通して出力
                    
                    1 今回 アマゾン で 上記 用紙 を 購入 して 満足 を して い ます 。 次回 も 注文 を し たい です 。 ←こういうのが渡される
                    1,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]                                                     ←こういう感じに変換される
                    
                    ここで行われているのがいわゆる順伝搬
                
                """
                #########################################################################################################################
                
                loss = criterion(outputs,label)
                """
                    ここで行われていのがいわゆる逆伝搬
                    データセットのラベルと予測したラベルを比較して損失を計算、ここでは二値分類なのでバイナリクロスエントロピーを使用
                    ミニバッチ毎に損失を計算している
                    epochの最後に平均を取る ??
                """
                #########################################################################################################################
                
                loss.backward()
                """
                    損失(正解との誤差)を元にモデルのパラメータを更新
                    不正解を学習率分だけしていった値を出して、正解に近づける
                    別で解説する
                """
                #########################################################################################################################
                
                optimizer.step()
                
                print(f"\rEpoch:{epoch+1} | Loss:{loss.item():.4f}",end="")
                
                # アーリーストッピングの呼び出し
                early_stopping(loss, model)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

                
            

                #OPTIMIZE:遅すぎるのでコメントアウト 
                #  # 損失をグラフに追加
                #  y_vals[global_step] = loss.item()
                #  with fig.batch_update():
                #     fig.data[0].y = y_vals
                #  global_step += 1
    
    
        #fig.write_html("loss.html")
        
        # 学習したモデルを保存
        #torch.save(model.state_dict(), "trained_model.pth")
       
    
    else:
        """
        評価
        """
        
        test_data,_ = load_data(TEST_DATA_PATH)#評価ではbatch_sizeは不要
        encoded_test_data = encode_texts(test_data, vocab)#学習用の語彙を使ってテストデータをエンコード
        
        # 学習済みのモデルをロード
        model = DoubleLSTMSentiment(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, OUTPUT_DIM)  
        model.load_state_dict(torch.load("checkpoint.pt"))
        
        #modelを評価モードにする
        model.eval()
        
        #テストデータの予測
        #勾配の更新を切る   
        with torch.no_grad():
            
            total = 0
            test_correct = 0
            
            for label,text in encoded_test_data:
                # 学習時と同じ処理を行う
                inputs = torch.tensor(text).unsqueeze(0)
                label_tensor = torch.tensor(label).unsqueeze(0)  # shapeを合わせるため
            
                outputs = model(inputs)
            
                # シグモイド関数を適用して0.5を閾値として二値化
                #答え
                predicted = (torch.sigmoid(outputs) > 0.5).float()
            
                total += 1
                test_correct += (predicted == label_tensor).sum().item()#予測と答えが一致しているか
        
        print(f"Accuracy:{100*test_correct/total}%")    


if __name__ == "__main__":
    main()