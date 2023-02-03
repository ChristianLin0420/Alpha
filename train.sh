
train_count=1

mixers='vdn qmix qtran qplex'
maps='3s_vs_5z 3s5z 5m_vs_6m 8m_vs_9m 10m_vs_11m 25m'

### 1c3s5z 2s3z 3m 8m 3s_vs_3z 
### 3s_vs_5z 3s5z 5m_vs_6m 8m_vs_9m 10m_vs_11m 25m 
### 3s5z_vs_3s6z 6h_vs_8z 27m_vs_30m bane_vs_bane corridor MMM2

for i in $(seq 1 $train_count); do 
    for mixer in $mixers; do
        for map in $maps; do
            python src/main.py --config=$mixer --env-config=sc2 --map_name=$map
        done
    done
done