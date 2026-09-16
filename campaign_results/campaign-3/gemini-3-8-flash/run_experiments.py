import os
import subprocess
import ast
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed


def run_eval(asset, days=3, eval_end="2026-08-15", time_increment=300, time_length=86400, env=None):
    cmd = [
        "uv", "run", "agent/predict.py",
        "--asset", asset,
        "--days", str(days),
        "--eval-end", eval_end,
        "--time-increment", str(time_increment),
        "--time-length", str(time_length),
        "--num-simulations", "1000"
    ]
    
    # Merge custom environment variables
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
        
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=run_env)
        if result.returncode != 0:
            print(f"Error running {cmd}: {result.stderr}")
            return None
            
        for line in result.stdout.splitlines():
            if line.startswith("summary:"):
                # Parse dict
                dict_str = line.split("summary:", 1)[1].strip()
                summary_dict = ast.literal_eval(dict_str)
                return summary_dict
        print(f"Summary not found in output: {result.stdout}")
        return None
    except Exception as e:
        print(f"Exception during run: {e}")
        return None


def worker(task):
    asset = task["asset"]
    comp = task["comp"]
    inc = task["inc"]
    length = task["len"]
    exp_name = task["exp_name"]
    env = task["env"].copy()
    
    # Generate a unique PRED_SUBDIR to avoid directory conflicts and race conditions
    exp_safe = exp_name.replace(" ", "_").replace("=", "_").lower()
    pred_subdir = f"exp_{asset}_{comp}_{exp_safe}".lower()
    env["PRED_SUBDIR"] = pred_subdir
    
    res = run_eval(
        asset=asset,
        time_increment=inc,
        time_length=length,
        env=env
    )
    if res:
        crps = res.get("mean_crps")
        score = res.get("final_smoothed_score")
        return {
            "Target": f"{asset}_{comp}",
            "Experiment": exp_name,
            "CRPS": crps,
            "Score": score,
            "Success": True
        }
    else:
        return {
            "Target": f"{asset}_{comp}",
            "Experiment": exp_name,
            "Success": False
        }


def main():
    print("Starting parallel hyperparameter tuning campaign...")
    
    # Let's define the experiments we want to run
    experiments = [
        # Baseline
        {"name": "Baseline", "env": {}},
        
        # Lookback Windows
        {"name": "Lookback 1d", "env": {"VOL_WINDOW_MINUTES": "1440"}},
        {"name": "Lookback 2d", "env": {"VOL_WINDOW_MINUTES": "2880"}},
        {"name": "Lookback 3d", "env": {"VOL_WINDOW_MINUTES": "4320"}},
        {"name": "Lookback 5d", "env": {"VOL_WINDOW_MINUTES": "7200"}},
        
        # EWMA Volatility (half-life in minutes)
        {"name": "EWMA HL 6h", "env": {"VOL_HALFLIFE_MINUTES": "360"}},
        {"name": "EWMA HL 12h", "env": {"VOL_HALFLIFE_MINUTES": "720"}},
        {"name": "EWMA HL 24h", "env": {"VOL_HALFLIFE_MINUTES": "1440"}},
        {"name": "EWMA HL 48h", "env": {"VOL_HALFLIFE_MINUTES": "2880"}},
        {"name": "EWMA HL 72h", "env": {"VOL_HALFLIFE_MINUTES": "4320"}},
        
        # Fat-tailed Shock Distributions
        {"name": "Student-t DF=3", "env": {"SHOCK_DISTRIBUTION": "student_t", "STUDENT_T_DF": "3.0"}},
        {"name": "Student-t DF=4", "env": {"SHOCK_DISTRIBUTION": "student_t", "STUDENT_T_DF": "4.0"}},
        {"name": "Student-t DF=5", "env": {"SHOCK_DISTRIBUTION": "student_t", "STUDENT_T_DF": "5.0"}},
        {"name": "Student-t DF=7", "env": {"SHOCK_DISTRIBUTION": "student_t", "STUDENT_T_DF": "7.0"}},
        {"name": "Student-t DF=10", "env": {"SHOCK_DISTRIBUTION": "student_t", "STUDENT_T_DF": "10.0"}},
        {"name": "Laplace", "env": {"SHOCK_DISTRIBUTION": "laplace"}},
    ]
    
    targets = [
        {"asset": "BTC", "comp": "CRYPTO_24H", "inc": 300, "len": 86400},
        {"asset": "BTC", "comp": "CRYPTO_1H", "inc": 60, "len": 3600},
        {"asset": "SP500", "comp": "COM_EQU_24H", "inc": 300, "len": 86400},
    ]
    
    tasks = []
    for t in targets:
        for exp in experiments:
            tasks.append({
                "asset": t["asset"],
                "comp": t["comp"],
                "inc": t["inc"],
                "len": t["len"],
                "exp_name": exp["name"],
                "env": exp["env"]
            })
            
    print(f"Total tasks to run: {len(tasks)}")
    
    results = []
    max_workers = 8  # Use 8 parallel workers to avoid overloading but achieve massive speedup
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, task): task for task in tasks}
        for future in as_completed(futures):
            res = future.result()
            task_info = futures[future]
            if res and res["Success"]:
                print(f"Finished: {res['Target']} - {res['Experiment']} | CRPS: {res['CRPS']:.4f}, Score: {res['Score']:.4f}")
                results.append(res)
            else:
                print(f"FAILED: {task_info['asset']}_{task_info['comp']} - {task_info['exp_name']}")
                
    # Tabulate results
    df = pd.DataFrame([r for r in results if r["Success"]])
    df.to_csv("agent/experiment_results.csv", index=False)
    
    print("\n=== EXPERIMENT RESULTS ===")
    for target in df["Target"].unique():
        print(f"\nResults for {target}:")
        sub_df = df[df["Target"] == target].copy()
        # Sort by CRPS ascending (lower is better)
        sub_df = sub_df.sort_values("CRPS")
        print(sub_df[["Experiment", "CRPS", "Score"]].to_string(index=False))


if __name__ == "__main__":
    main()
