# OBNC REMOVE SENSITIVE

python run_noiseinjection.py bank housing OBNC-remove-sensitive random 
python run_noiseinjection.py phishing having_IP_Address OBNC-remove-sensitive random
python run_noiseinjection.py monks1 attr3 OBNC-remove-sensitive random  
python run_noiseinjection.py biodeg V25 OBNC-remove-sensitive random 
python run_noiseinjection.py credit A1 OBNC-remove-sensitive random
python run_noiseinjection.py sick referral_source_other OBNC-remove-sensitive random
python run_noiseinjection.py churn voice_mail_plan OBNC-remove-sensitive random
python run_noiseinjection.py vote immigration OBNC-remove-sensitive random  
python run_noiseinjection.py ads local OBNC-remove-sensitive random

python run_noiseinjection.py bank housing OBNC-remove-sensitive flip 
python run_noiseinjection.py phishing having_IP_Address OBNC-remove-sensitive flip
python run_noiseinjection.py monks1 attr3 OBNC-remove-sensitive flip  
python run_noiseinjection.py biodeg V25 OBNC-remove-sensitive flip 
python run_noiseinjection.py credit A1 OBNC-remove-sensitive flip
python run_noiseinjection.py sick referral_source_other OBNC-remove-sensitive flip
python run_noiseinjection.py churn voice_mail_plan OBNC-remove-sensitive flip
python run_noiseinjection.py vote immigration OBNC-remove-sensitive flip  
python run_noiseinjection.py ads local OBNC-remove-sensitive flip

python run_noiseinjection.py bank housing OBNC-remove-sensitive bias 
python run_noiseinjection.py phishing having_IP_Address OBNC-remove-sensitive bias
python run_noiseinjection.py monks1 attr3 OBNC-remove-sensitive bias  
python run_noiseinjection.py biodeg V25 OBNC-remove-sensitive bias 
python run_noiseinjection.py credit A1 OBNC-remove-sensitive bias
python run_noiseinjection.py sick referral_source_other OBNC-remove-sensitive bias
python run_noiseinjection.py churn voice_mail_plan OBNC-remove-sensitive bias
python run_noiseinjection.py vote immigration OBNC-remove-sensitive bias  
python run_noiseinjection.py ads local OBNC-remove-sensitive bias

python run_noiseinjection.py bank housing OBNC-remove-sensitive balanced_bias 
python run_noiseinjection.py phishing having_IP_Address OBNC-remove-sensitive balanced_bias
python run_noiseinjection.py monks1 attr3 OBNC-remove-sensitive balanced_bias  
python run_noiseinjection.py biodeg V25 OBNC-remove-sensitive balanced_bias 
python run_noiseinjection.py credit A1 OBNC-remove-sensitive balanced_bias
python run_noiseinjection.py sick referral_source_other OBNC-remove-sensitive balanced_bias
python run_noiseinjection.py churn voice_mail_plan OBNC-remove-sensitive balanced_bias
python run_noiseinjection.py vote immigration OBNC-remove-sensitive balanced_bias  
python run_noiseinjection.py ads local OBNC-remove-sensitive balanced_bias

# OBNC OPTIMIZE DEMOGRAPHIC PARITY

python run_noiseinjection.py bank housing OBNC-optimize-demographic-parity random 
python run_noiseinjection.py phishing having_IP_Address OBNC-optimize-demographic-parity random
python run_noiseinjection.py monks1 attr3 OBNC-optimize-demographic-parity random  
python run_noiseinjection.py biodeg V25 OBNC-optimize-demographic-parity random 
python run_noiseinjection.py credit A1 OBNC-optimize-demographic-parity random
python run_noiseinjection.py sick referral_source_other OBNC-optimize-demographic-parity random
python run_noiseinjection.py churn voice_mail_plan OBNC-optimize-demographic-parity random
python run_noiseinjection.py vote immigration OBNC-optimize-demographic-parity random  
python run_noiseinjection.py ads local OBNC-optimize-demographic-parity random

python run_noiseinjection.py bank housing OBNC-optimize-demographic-parity flip 
python run_noiseinjection.py phishing having_IP_Address OBNC-optimize-demographic-parity flip
python run_noiseinjection.py monks1 attr3 OBNC-optimize-demographic-parity flip  
python run_noiseinjection.py biodeg V25 OBNC-optimize-demographic-parity flip 
python run_noiseinjection.py credit A1 OBNC-optimize-demographic-parity flip
python run_noiseinjection.py sick referral_source_other OBNC-optimize-demographic-parity flip
python run_noiseinjection.py churn voice_mail_plan OBNC-optimize-demographic-parity flip
python run_noiseinjection.py vote immigration OBNC-optimize-demographic-parity flip  
python run_noiseinjection.py ads local OBNC-optimize-demographic-parity flip

python run_noiseinjection.py bank housing OBNC-optimize-demographic-parity bias 
python run_noiseinjection.py phishing having_IP_Address OBNC-optimize-demographic-parity bias
python run_noiseinjection.py monks1 attr3 OBNC-optimize-demographic-parity bias  
python run_noiseinjection.py biodeg V25 OBNC-optimize-demographic-parity bias 
python run_noiseinjection.py credit A1 OBNC-optimize-demographic-parity bias
python run_noiseinjection.py sick referral_source_other OBNC-optimize-demographic-parity bias
python run_noiseinjection.py churn voice_mail_plan OBNC-optimize-demographic-parity bias
python run_noiseinjection.py vote immigration OBNC-optimize-demographic-parity bias  
python run_noiseinjection.py ads local OBNC-optimize-demographic-parity bias

python run_noiseinjection.py bank housing OBNC-optimize-demographic-parity balanced_bias 
python run_noiseinjection.py phishing having_IP_Address OBNC-optimize-demographic-parity balanced_bias
python run_noiseinjection.py monks1 attr3 OBNC-optimize-demographic-parity balanced_bias  
python run_noiseinjection.py biodeg V25 OBNC-optimize-demographic-parity balanced_bias 
python run_noiseinjection.py credit A1 OBNC-optimize-demographic-parity balanced_bias
python run_noiseinjection.py sick referral_source_other OBNC-optimize-demographic-parity balanced_bias
python run_noiseinjection.py churn voice_mail_plan OBNC-optimize-demographic-parity balanced_bias
python run_noiseinjection.py vote immigration OBNC-optimize-demographic-parity balanced_bias  
python run_noiseinjection.py ads local OBNC-optimize-demographic-parity balanced_bias

# OBNC OPTIMIZE DEMOGRAPHIC PARITY PROB=0.5

python run_noiseinjection.py bank housing OBNC-optimize-demographic-parity random --prob 0.5
python run_noiseinjection.py phishing having_IP_Address OBNC-optimize-demographic-parity random --prob 0.5
python run_noiseinjection.py monks1 attr3 OBNC-optimize-demographic-parity random --prob 0.5
python run_noiseinjection.py biodeg V25 OBNC-optimize-demographic-parity random --prob 0.5
python run_noiseinjection.py credit A1 OBNC-optimize-demographic-parity random --prob 0.5
python run_noiseinjection.py sick referral_source_other OBNC-optimize-demographic-parity random --prob 0.5
python run_noiseinjection.py churn voice_mail_plan OBNC-optimize-demographic-parity random --prob 0.5
python run_noiseinjection.py vote immigration OBNC-optimize-demographic-parity random --prob 0.5
python run_noiseinjection.py ads local OBNC-optimize-demographic-parity random --prob 0.5

python run_noiseinjection.py bank housing OBNC-optimize-demographic-parity flip --prob 0.5
python run_noiseinjection.py phishing having_IP_Address OBNC-optimize-demographic-parity flip --prob 0.5
python run_noiseinjection.py monks1 attr3 OBNC-optimize-demographic-parity flip --prob 0.5
python run_noiseinjection.py biodeg V25 OBNC-optimize-demographic-parity flip --prob 0.5
python run_noiseinjection.py credit A1 OBNC-optimize-demographic-parity flip --prob 0.5
python run_noiseinjection.py sick referral_source_other OBNC-optimize-demographic-parity flip --prob 0.5
python run_noiseinjection.py churn voice_mail_plan OBNC-optimize-demographic-parity flip --prob 0.5
python run_noiseinjection.py vote immigration OBNC-optimize-demographic-parity flip --prob 0.5
python run_noiseinjection.py ads local OBNC-optimize-demographic-parity flip --prob 0.5

python run_noiseinjection.py bank housing OBNC-optimize-demographic-parity bias --prob 0.5
python run_noiseinjection.py phishing having_IP_Address OBNC-optimize-demographic-parity bias --prob 0.5
python run_noiseinjection.py monks1 attr3 OBNC-optimize-demographic-parity bias --prob 0.5
python run_noiseinjection.py biodeg V25 OBNC-optimize-demographic-parity bias --prob 0.5
python run_noiseinjection.py credit A1 OBNC-optimize-demographic-parity bias --prob 0.5
python run_noiseinjection.py sick referral_source_other OBNC-optimize-demographic-parity bias --prob 0.5
python run_noiseinjection.py churn voice_mail_plan OBNC-optimize-demographic-parity bias --prob 0.5
python run_noiseinjection.py vote immigration OBNC-optimize-demographic-parity bias --prob 0.5
python run_noiseinjection.py ads local OBNC-optimize-demographic-parity bias --prob 0.5

python run_noiseinjection.py bank housing OBNC-optimize-demographic-parity balanced_bias --prob 0.5
python run_noiseinjection.py phishing having_IP_Address OBNC-optimize-demographic-parity balanced_bias --prob 0.5
python run_noiseinjection.py monks1 attr3 OBNC-optimize-demographic-parity balanced_bias --prob 0.5
python run_noiseinjection.py biodeg V25 OBNC-optimize-demographic-parity balanced_bias --prob 0.5
python run_noiseinjection.py credit A1 OBNC-optimize-demographic-parity balanced_bias --prob 0.5
python run_noiseinjection.py sick referral_source_other OBNC-optimize-demographic-parity balanced_bias --prob 0.5
python run_noiseinjection.py churn voice_mail_plan OBNC-optimize-demographic-parity balanced_bias --prob 0.5
python run_noiseinjection.py vote immigration OBNC-optimize-demographic-parity balanced_bias --prob 0.5
python run_noiseinjection.py ads local OBNC-optimize-demographic-parity balanced_bias --prob 0.5

# OBNC FAIR

python run_noiseinjection.py bank housing OBNC-fair random 
python run_noiseinjection.py phishing having_IP_Address OBNC-fair random
python run_noiseinjection.py monks1 attr3 OBNC-fair random  
python run_noiseinjection.py biodeg V25 OBNC-fair random 
python run_noiseinjection.py credit A1 OBNC-fair random
python run_noiseinjection.py sick referral_source_other OBNC-fair random
python run_noiseinjection.py churn voice_mail_plan OBNC-fair random
python run_noiseinjection.py vote immigration OBNC-fair random  
python run_noiseinjection.py ads local OBNC-fair random

python run_noiseinjection.py bank housing OBNC-fair flip 
python run_noiseinjection.py phishing having_IP_Address OBNC-fair flip
python run_noiseinjection.py monks1 attr3 OBNC-fair flip  
python run_noiseinjection.py biodeg V25 OBNC-fair flip 
python run_noiseinjection.py credit A1 OBNC-fair flip
python run_noiseinjection.py sick referral_source_other OBNC-fair flip
python run_noiseinjection.py churn voice_mail_plan OBNC-fair flip
python run_noiseinjection.py vote immigration OBNC-fair flip  
python run_noiseinjection.py ads local OBNC-fair flip

python run_noiseinjection.py bank housing OBNC-fair bias 
python run_noiseinjection.py phishing having_IP_Address OBNC-fair bias
python run_noiseinjection.py monks1 attr3 OBNC-fair bias  
python run_noiseinjection.py biodeg V25 OBNC-fair bias 
python run_noiseinjection.py credit A1 OBNC-fair bias
python run_noiseinjection.py sick referral_source_other OBNC-fair bias
python run_noiseinjection.py churn voice_mail_plan OBNC-fair bias
python run_noiseinjection.py vote immigration OBNC-fair bias  
python run_noiseinjection.py ads local OBNC-fair bias

python run_noiseinjection.py bank housing OBNC-fair balanced_bias 
python run_noiseinjection.py phishing having_IP_Address OBNC-fair balanced_bias
python run_noiseinjection.py monks1 attr3 OBNC-fair balanced_bias  
python run_noiseinjection.py biodeg V25 OBNC-fair balanced_bias 
python run_noiseinjection.py credit A1 OBNC-fair balanced_bias
python run_noiseinjection.py sick referral_source_other OBNC-fair balanced_bias
python run_noiseinjection.py churn voice_mail_plan OBNC-fair balanced_bias
python run_noiseinjection.py vote immigration OBNC-fair balanced_bias  
python run_noiseinjection.py ads local OBNC-fair balanced_bias

# OBNC FAIR PROB=0.5

python run_noiseinjection.py bank housing OBNC-fair random --prob 0.5
python run_noiseinjection.py phishing having_IP_Address OBNC-fair random --prob 0.5
python run_noiseinjection.py monks1 attr3 OBNC-fair random  --prob 0.5
python run_noiseinjection.py biodeg V25 OBNC-fair random --prob 0.5
python run_noiseinjection.py credit A1 OBNC-fair random --prob 0.5
python run_noiseinjection.py sick referral_source_other OBNC-fair random --prob 0.5
python run_noiseinjection.py churn voice_mail_plan OBNC-fair random --prob 0.5
python run_noiseinjection.py vote immigration OBNC-fair random --prob 0.5
python run_noiseinjection.py ads local OBNC-fair random --prob 0.5

python run_noiseinjection.py bank housing OBNC-fair flip --prob 0.5 
python run_noiseinjection.py phishing having_IP_Address OBNC-fair flip --prob 0.5
python run_noiseinjection.py monks1 attr3 OBNC-fair flip --prob 0.5
python run_noiseinjection.py biodeg V25 OBNC-fair flip --prob 0.5
python run_noiseinjection.py credit A1 OBNC-fair flip --prob 0.5
python run_noiseinjection.py sick referral_source_other OBNC-fair flip --prob 0.5
python run_noiseinjection.py churn voice_mail_plan OBNC-fair flip --prob 0.5
python run_noiseinjection.py vote immigration OBNC-fair flip --prob 0.5
python run_noiseinjection.py ads local OBNC-fair flip --prob 0.5

python run_noiseinjection.py bank housing OBNC-fair bias --prob 0.5
python run_noiseinjection.py phishing having_IP_Address OBNC-fair bias --prob 0.5
python run_noiseinjection.py monks1 attr3 OBNC-fair bias --prob 0.5
python run_noiseinjection.py biodeg V25 OBNC-fair bias --prob 0.5
python run_noiseinjection.py credit A1 OBNC-fair bias --prob 0.5
python run_noiseinjection.py sick referral_source_other OBNC-fair bias --prob 0.5
python run_noiseinjection.py churn voice_mail_plan OBNC-fair bias --prob 0.5
python run_noiseinjection.py vote immigration OBNC-fair bias --prob 0.5
python run_noiseinjection.py ads local OBNC-fair bias --prob 0.5

python run_noiseinjection.py bank housing OBNC-fair balanced_bias --prob 0.5 
python run_noiseinjection.py phishing having_IP_Address OBNC-fair balanced_bias --prob 0.5
python run_noiseinjection.py monks1 attr3 OBNC-fair balanced_bias --prob 0.5
python run_noiseinjection.py biodeg V25 OBNC-fair balanced_bias --prob 0.5
python run_noiseinjection.py credit A1 OBNC-fair balanced_bias --prob 0.5
python run_noiseinjection.py sick referral_source_other OBNC-fair balanced_bias --prob 0.5
python run_noiseinjection.py churn voice_mail_plan OBNC-fair balanced_bias --prob 0.5
python run_noiseinjection.py vote immigration OBNC-fair balanced_bias --prob 0.5
python run_noiseinjection.py ads local OBNC-fair balanced_bias --prob 0.5