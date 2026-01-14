import pandas as pd

# Define column names (based on NSL-KDD dataset documentation)
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
           "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
           "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
           "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
           "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
           "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
           "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
           "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
           "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
           "dst_host_serror_rate", "dst_host_srv_serror_rate",
           "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]

# Convert train dataset
train_txt_path = "data/KDDTrain+.txt"
train_csv_path = "data/NSL_KDD_train.csv"

df_train = pd.read_csv(train_txt_path, names=columns)
df_train.to_csv(train_csv_path, index=False)
print(f"Converted {train_txt_path} to {train_csv_path}")

# Convert test dataset
test_txt_path = "data/KDDTest+.txt"
test_csv_path = "data/NSL_KDD_test.csv"

df_test = pd.read_csv(test_txt_path, names=columns)
df_test.to_csv(test_csv_path, index=False)
print(f"Converted {test_txt_path} to {test_csv_path}")
