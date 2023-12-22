script_name=$1
script_full_path="scripts.$script_name"

./stop_script.sh $1

# Log file
logs_dir="logs"
log_file_name=${script_name//./_}
log_file="$logs_dir/$log_file_name.log"
#mkdir -p $logs_dir


python -u -m $script_full_path #> $log_file 2>&1 &
