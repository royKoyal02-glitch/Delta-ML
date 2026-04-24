import rdkit
import os
import shutil
import re
import subprocess
import time
from datetime import datetime
from rdkit import Chem
from rdkit.Chem import AllChem

# === Config ===
USER = "r.koyal"
MAX_JOBS = 10
ROOT_DIR = "."
RUN_SCRIPT_NAME = "run.sh"
NPROC = 16
MEM = "16GB"
TITLE = "Title Card Required"
def get_run_mode():
    print("\nSelect workflow mode:")
    print("1. Optimization only")
    print("2. Optimization + TDDFT")
    while True:
        choice = input("Enter choice [1/2]: ").strip()
        if choice in ['1', '2']:
            return int(choice)
        print("[ERROR] Invalid choice. Please enter 1 or 2.")
def get_user_inputs():
    # Ask if xTB optimization should be performed
    do_xtb = input("Do xTB optimization before Gaussian? (y/n): ").strip().lower() == 'y'

    # Total charge
    while True:
        try:
            charge = int(input("Enter total molecular charge [default 0]: ").strip() or 0)
            break
        except ValueError:
            print("[ERROR] Invalid number. Please try again.")

    # Multiplicity
    while True:
        try:
            multiplicity = int(input("Enter multiplicity (1=Singlet, 2=Doublet, 3=Triplet, etc.) [default 1]: ").strip() or 1)
            if multiplicity >= 1:
                break
            else:
                print("[ERROR] Multiplicity must be >= 1")
        except ValueError:
            print("[ERROR] Invalid number. Please try again.")

    # Functional
    functional = input("Enter Gaussian functional (default b3lyp): ").strip() or "b3lyp"

    # Basis set
    basis = input("Enter Gaussian basis set (default 6-31g(d,p)): ").strip() or "6-31g(d,p)"

    print(f"[INFO] Settings -> xTB: {do_xtb}, Charge: {charge}, Multiplicity: {multiplicity}, Gaussian: {functional}/{basis}")
    return do_xtb, charge, multiplicity, functional, basis


def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")
def check_requirements():
    gjfs = [f for f in os.listdir(ROOT_DIR) if f.endswith('.gjf')]
    run_path = os.path.join(ROOT_DIR, RUN_SCRIPT_NAME)
    if not gjfs:
        log("[ERROR] No .gjf files found.")
        return False
    if not os.path.isfile(run_path):
        log(f"[ERROR] {RUN_SCRIPT_NAME} missing.")
        return False
    log(f"[CHECK] Found {len(gjfs)} .gjf files and {RUN_SCRIPT_NAME}.")
    return True

def convert_smiles_to_gjf(smiles, filename, charge=0, multiplicity=1):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    mol = Chem.AddHs(mol)

    if AllChem.EmbedMolecule(mol) != 0:
        raise RuntimeError("3D embedding failed.")
    AllChem.UFFOptimizeMolecule(mol)

    if not filename.endswith(".gjf"):
        filename += ".gjf"

    with open(filename, 'w') as f:
        chk_name = filename.rsplit(".", 1)[0]
        f.write(f"%chk={chk_name}.chk\n")
        f.write("# opt freq b3lyp/6-31G(d,p)\n\n")
        f.write("Generated from SMILES\n\n")
        f.write(f"{charge} {multiplicity}\n")   # <-- Charge and multiplicity
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol():<2}   {pos.x:.6f}   {pos.y:.6f}   {pos.z:.6f}\n")
        f.write("\n")

    print(f"[OK] Wrote {filename} from SMILES with charge={charge}, multiplicity={multiplicity}.")


#-----------XTB PART------------#
def run_xtb_optimization_with_crest(gjf_path, charge=0, multiplicity=1):
    base = os.path.splitext(os.path.basename(gjf_path))[0]
    xtb_dir = os.path.join(base, "xtbopt")
    os.makedirs(xtb_dir, exist_ok=True)

    # --- Convert gjf to xyz ---#
    xyz_path = os.path.join(xtb_dir, f"{base}.xyz")
    with open(gjf_path) as f:
        lines = f.readlines()

    atoms = []
    for line in lines:
        if re.match(r'^\s*[A-Z][a-z]?\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+', line):
            atoms.append(line.strip())

    with open(xyz_path, 'w') as f:
        f.write(f"{len(atoms)}\n\n")
        for atom in atoms:
            f.write(atom + "\n")

    # --- Spin mapping --- #
    uhf = max(0, multiplicity - 1)  # singlet=0, doublet=1, triplet=2

    # --- Run xTB optimization ---#
    try:
        subprocess.run(
            ["xtb", f"{base}.xyz", "--opt", "--gfn2",
             "--chrg", str(charge), "--uhf", str(uhf)],
            cwd=xtb_dir, check=True
        )
    except subprocess.CalledProcessError:
        log(f"[ERROR] xTB optimization failed for {base}")
        return None

    # --- Run CREST conformer search ---#
    try:
        subprocess.run(
            ["crest", "xtbopt.xyz", "--gfn2",
             "--chrg", str(charge), "--uhf", str(uhf)],
            cwd=xtb_dir, check=True
        )
    except subprocess.CalledProcessError:
        log(f"[ERROR] CREST failed for {base}")
        return None

    crest_opt = os.path.join(xtb_dir, "crest_conformers.xyz")
    if not os.path.isfile(crest_opt):
        log(f"[ERROR] CREST output not found: {crest_opt}")
        return None

    # --- Extract last geometry from CREST --- #
    with open(crest_opt, 'r') as f:
        lines = f.readlines()

    frames, frame = [], []
    for line in lines:
        if re.match(r'^\d+$', line.strip()):
            if frame:
                frames.append(frame)
                frame = []
            frame.append(line)
        else:
            frame.append(line)
    if frame:
        frames.append(frame)

    final_frame = frames[-1][2:]  # skip atom count + comment
    coords = [l.strip() for l in final_frame if len(l.strip().split()) == 4]

    # --- Replace geometry in gjf ---
    with open(gjf_path) as f:
        old_lines = f.readlines()

    new_lines = []
    coord_block = False
    for line in old_lines:
        if re.match(r'^\s*[A-Z][a-z]?\s+-?\d+\.\d+\s+-?\d+\.\d+\s+-?\d+\.\d+', line):
            if not coord_block:
                new_lines.extend(coords)
                coord_block = True
            continue
        new_lines.append(line.rstrip())

    with open(gjf_path, 'w') as f:
        f.write('\n'.join(new_lines) + '\n')

    log(f"[OK] Geometry replaced using CREST-optimized structure for {base} "
        f"(charge={charge}, multiplicity={multiplicity}, uhf={uhf})")
    return gjf_path



        
def get_user_choices():
    func = input("Enter functional (default b3lyp): ").strip() or "b3lyp"
    basis = input("Enter basis (default 6-31g(d,p)): ").strip() or "6-31g(d,p)"
    log(f"Using {func}/{basis}")
    return func, basis
def prepare_com_files(func, basis,charge, multiplicity, mode_name):
    
    for file in os.listdir(ROOT_DIR):
        if file.endswith(".gjf"):
            mol_name = os.path.splitext(file)[0]
            src_path = os.path.join(ROOT_DIR, file)

            # Create folder for this molecule and multiplicity
            dest_dir = os.path.join(ROOT_DIR, mol_name, mode_name)
            os.makedirs(dest_dir, exist_ok=True)

            # Read geometry lines from source .gjf
            coords = []
            read_atoms = False
            with open(src_path, "r") as f:
                for line in f:
                    line = line.strip()
                    # Start reading after charge/multiplicity line
                    if read_atoms:
                        if line == "" or line[0].isdigit():
                            break
                        coords.append(line)
                    elif line.startswith("0 ") or line.startswith("1 ") or line.startswith("2 ") or line.startswith("3 "):
                        read_atoms = True
                        coords.append(line)  # This line contains charge & multiplicity first
                        
            # Prepare .com file path
            com_file = os.path.join(dest_dir, mol_name + ".com")

            # Write Gaussian .com file cleanly
            with open(com_file, "w") as out:
                out.write(f"%chk={mol_name}.chk\n")
                out.write(f"%nprocshared={NPROC}\n")
                out.write(f"%mem={MEM}\n")
                out.write(f"# opt freq {func}/{basis}\n\n")
                out.write(f"{TITLE}\n\n")
                out.write(f"{charge} {multiplicity}\n")  # Charge fixed to 0, multiplicity dynamic

                # Write only coordinates, no connectivity
                for coord in coords:
                    if coord.startswith("0 ") or coord.startswith("1 ") or coord.startswith("2 ") or coord.startswith("3 "):
                        continue  # skip charge/multiplicity line if captured
                    out.write(coord + "\n")

                out.write("\n")

            log(f"[INFO] Prepared {com_file} with multiplicity {multiplicity}")

def copy_run_script(mode_name):
    for root, dirs, files in os.walk(ROOT_DIR):
        if mode_name in dirs:
            folder_path = os.path.join(root, mode_name)
            dest_file = os.path.join(folder_path, RUN_SCRIPT_NAME)

            if not os.path.isfile(RUN_SCRIPT_NAME):
                log("[ERROR] Template run.sh missing in ROOT_DIR.")
                return

            shutil.copy(RUN_SCRIPT_NAME, dest_file)
            log(f"[COPY] run.sh added to {folder_path}")
                                                            

def update_jobnames():
    for root, _, files in os.walk(ROOT_DIR):
        if RUN_SCRIPT_NAME in files:
            com = next((f for f in files if f.endswith('.com')), None)
            if not com: continue
            base = os.path.splitext(com)[0]
            rp = os.path.join(root, RUN_SCRIPT_NAME)
            lines = open(rp).readlines()
            for i, L in enumerate(lines):
                if "#SBATCH --job-name=" in L:
                    lines[i] = f"#SBATCH --job-name={base}\n"
                if "JobFile=" in L:
                    lines[i] = f"JobFile={base}\n"
            open(rp, 'w').writelines(lines)
            log(f"Updated names in {rp}")
def submit_optimization_jobs(mode_name):
    submitted_jobs = {}

    # Find all directories for the selected multiplicity
    jdirs = [os.path.join(d, mode_name) for d in os.listdir(ROOT_DIR)
             if os.path.isdir(os.path.join(d, mode_name))]
    
    queue = sorted(jdirs)
    log(f"[INFO] Found {len(queue)} {mode_name} jobs to submit.")

    while queue:
        cnt = get_job_count(USER)
        user_jobs = get_user_jobs(USER)

        if cnt < MAX_JOBS:
            jd = queue.pop(0)

            # Skip if job already finished-----------
            logs = [f for f in os.listdir(jd) if f.endswith(".log")]
            if logs and is_job_done(os.path.join(jd, logs[0])):
                log(f"[SKIP] done: {jd}")
                continue

            # Skip if already running--------------------
            if is_running(jd, user_jobs):
                log(f"[WAIT] running: {jd}")
                continue

            # Submit job if run.sh exists------------------
            run_path = os.path.join(jd, RUN_SCRIPT_NAME)
            if os.path.isfile(run_path):
                result = subprocess.run(
                    ["sbatch", RUN_SCRIPT_NAME],
                    cwd=jd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                if result.returncode == 0:
                    m = re.search(r"Submitted batch job (\d+)", result.stdout)
                    if m:
                        jobid = m.group(1)
                        submitted_jobs[jd] = jobid
                        log(f"[SUBMIT] {jd}: jobid={jobid}")
                    else:
                        log(f"[ERROR] Could not parse jobid: {result.stdout.strip()}")
                else:
                    log(f"[ERROR] Failed to submit {jd}: {result.stderr.strip()}")
            else:
                log(f"[MISSING] run.sh in {jd}")
        else:
            log(f"[WAIT] Job limit reached ({cnt}/{MAX_JOBS}). Sleeping 60s...")
            time.sleep(60)

    log(f"[DONE] All {mode_name} jobs submitted or skipped.")
    return submitted_jobs

def is_job_done(logfile):
    if not os.path.isfile(logfile):
        return False
    text = open(logfile, errors="ignore").read()
    return "Normal termination" in text
def check_negative_freq(path):
    try:
        with open(path, errors="ignore") as f:
            text = f.read()
        line_re = re.compile(r"^\s*Frequencies\s+--\s+(.+)$", re.MULTILINE)
        num_re = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")

        for m in line_re.finditer(text):
            vals = [float(x) for x in num_re.findall(m.group(1))]
            for v in vals:
                if v < 0:
                    return True, v  # Found a negative (imaginary) frequency
    except Exception:
        pass

    return False, None
def check_log_status(logfile):
    if not os.path.isfile(logfile):
        return "missing"
    text = open(logfile, errors="ignore").read()
    if "Normal termination" not in text:
        return "abnormal"
    neg, v = check_negative_freq(logfile)
    if neg:
        return "negfreq"
    return "ok"

def get_user_jobs(user):
    try:
        out = subprocess.check_output(f"squeue -u {user}", shell=True).decode()
        return out.strip().splitlines()[1:]
    except:
        return []

def get_job_count(user):
    try:
        out = subprocess.check_output(f"squeue -u {user} -h -o '%i'", shell=True).decode()
        return len(out.splitlines())
    except:
        return 0

def is_running(job_dir, user_jobs):
    job_name = os.path.basename(os.path.dirname(job_dir)) if job_dir.endswith("singlet") else os.path.basename(job_dir)
    return any(job_name in line for line in user_jobs)

def is_running_jobid(jobid):
    try:
        out = subprocess.check_output(
            f"squeue -j {jobid} -h -o '%i'", shell=True
        ).decode().strip()
        return out == jobid
    except:
        return False

def scan_logs():
    bad = {}
    for r, _, files in os.walk(ROOT_DIR):
        for f in files:
            if f.endswith('.log'):
                full = os.path.join(r, f)
                neg, v = check_negative_freq(full)
                if neg:
                    bad[r] = v
    return bad
def extract_last_scf_energy(log_path):
    try:
        with open(log_path, 'r', errors='ignore') as file:
            last_energy = None
            for line in file:
                if "SCF Done" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "=":
                            last_energy = float(parts[i + 1])
                            break
        return last_energy
    except:
        return None
def wait_for_optimization_completion(submitted_jobs, mode_name, max_wait_minutes=360000, check_interval=60):
    import pandas as pd
    log(f"[INFO] Waiting for {mode_name} optimization jobs to complete...")

    start_time = time.time()
    still_running = dict(submitted_jobs)
    results = []

    while True:
        finished = []

        # --- Check each job ---
        for jd, jobid in still_running.items():
            if is_running_jobid(jobid):
                continue  # Still running

            logs = [f for f in os.listdir(jd) if f.endswith(".log")]
            if not logs:
                log(f"[ERROR] No log file found for {jd}")
                finished.append(jd)
                continue

            log_path = os.path.join(jd, logs[0])
            status = check_log_status(log_path)

            # Extract SCF Done energy if the job finished normally
            if status in ["ok", "negfreq"]:
                energy = extract_last_scf_energy(log_path)
                if energy is not None:
                    results.append({
                        "Molecule": os.path.basename(os.path.dirname(jd)),
                        "Folder": jd,
                        "Last SCF Energy (Hartree)": energy,
                        "Status": status
                    })
                    log(f"[INFO] {jd} -> Energy: {energy:.6f} Hartree, Status: {status}")
            else:
                log(f"[WARN] {jd}: Job finished with status '{status}'")

            finished.append(jd)

        # Remove completed jobs from the running list
        for jd in finished:
            still_running.pop(jd, None)

        # --- If all jobs finished ---
        if not still_running:
            if results:
                pd.DataFrame(results).to_csv("Optimization_Energies.csv", index=False)
                log("[SAVED] Optimization energies saved to Optimization_Energies.csv")
            else:
                log("[WARNING] No SCF energies collected. Check log files.")

            # Final negative frequency check
            bads = scan_logs()
            if bads:
                log("[WARNING] Negative frequencies found:")
                for r, v in bads.items():
                    log(f"  - {r}: {v}")
            else:
                log("[OK] No negative frequencies detected.")
            return

        # --- Timeout check ---
        elapsed_minutes = (time.time() - start_time) / 60
        if elapsed_minutes > max_wait_minutes:
            log("[WARNING] Timeout reached. Some jobs still running:")
            for jd, jobid in still_running.items():
                log(f"  - {jd} (jobid {jobid})")
            return

        # --- Wait before checking again ---
        log(f"[WAIT] {len(still_running)} {mode_name} jobs still running. "
            f"Checking again in {check_interval} seconds...")
        time.sleep(check_interval)

def atno2sym(Z):
    periodic_table = {
        1: "H", 2: "He",
        3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
        11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
        19: "K", 20: "Ca", 21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn",
        26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
        31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
        37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
        44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd",
        49: "In", 50: "Sn", 51: "Sb", 52: "Te", 53: "I", 54: "Xe",
        55: "Cs", 56: "Ba",
        57: "La", 58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu",
        64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
        72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt",
        79: "Au", 80: "Hg",
        81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn",
        87: "Fr", 88: "Ra",
        89: "Ac", 90: "Th", 91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am",
        96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm", 101: "Md",
        102: "No", 103: "Lr",
        104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt",
        110: "Ds", 111: "Rg", 112: "Cn",
        113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
    }
    return periodic_table.get(Z, "X")
#def atno2sym(Z):
   # return {
       # 1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
       # 16: "S", 17: "Cl", 35: "Br", 53: "I"
   # }.get(Z, "X")

def extract_geom(log):
    L = open(log).read().splitlines()
    for i in reversed(range(len(L))):
        if "Standard orientation" in L[i]:
            s = i + 5
            break
    else:
        raise ValueError("No geometry found in log file.")
    geom = []
    for L2 in L[s:]:
        if "-----" in L2 or not L2.strip():
            break
        parts = L2.split()
        geom.append(f"{atno2sym(int(parts[1])):<2}  {parts[3]}  {parts[4]}  {parts[5]}")
    return geom
def create_tddft(mol_dir, log_path, mol_name, func, basis, charge, multiplicity):
    td_dir = os.path.join(mol_dir, "tddft")
    os.makedirs(td_dir, exist_ok=True)

    com_file = os.path.join(td_dir, f"opt-{mol_name}.com")
    chk_file = f"opt-{mol_name}.chk"

    if os.path.isfile(com_file):
        log(f"[SKIP] TDDFT file already exists: {com_file}")
        return

    try:
        geom = extract_geom(log_path)  # <-- changed from log to log_path
    except Exception as e:
        log(f"[ERROR] Geometry extraction failed for {log_path}: {e}")
        return

    with open(com_file, 'w') as f:
        f.write(f"%chk={chk_file}\n")
        f.write(f"%nprocshared={NPROC}\n")
        f.write(f"%mem={MEM}\n")
        f.write(f"# td(nstates=10,50-50) {func}/{basis}\n\n")
        f.write(f"{TITLE}\n\n")
        f.write(f"{charge} {multiplicity}\n")
        for line in geom:
            f.write(line + "\n")
        f.write("\n")

    log(f"[CREATED] TDDFT input: {com_file}")


def _negfreq_molecule_set(bads, root=ROOT_DIR):
    bad_mols = set()
    for bad_dir in bads.keys():
        rel = os.path.relpath(bad_dir, root)
        top = rel.split(os.sep)[0]
        if top:
            bad_mols.add(top)
    return bad_mols
def gen_tddft(bads, func, basis, charge, multiplicity):
    bad_mols = _negfreq_molecule_set(bads, ROOT_DIR)

    for mol_dir in os.listdir(ROOT_DIR):
        if not os.path.isdir(mol_dir):
            continue

        # Skip molecules with negative frequencies
        if mol_dir in bad_mols:
            log(f"[SKIP] TDDFT for {mol_dir} due to negative frequency.")
            continue

        # Look for optimization logs in multiplicity folders
        found_log = None
        for sub_dir in os.listdir(mol_dir):
            sub_path = os.path.join(mol_dir, sub_dir)
            if not os.path.isdir(sub_path) or sub_dir == "tddft":
                continue

            # Find Gaussian log inside this multiplicity folder
            for fn in os.listdir(sub_path):
                if fn.endswith(".log"):
                    log_file = os.path.join(sub_path, fn)
                    with open(log_file, "r", errors="ignore") as f:
                        if "Normal termination" in f.read():
                            found_log = log_file
                            break
            if found_log:
                break  # stop at the first completed optimization

        if not found_log:
            log(f"[SKIP] {mol_dir} (no completed optimization found)")
            continue

        # Generate TDDFT input in top-level tddft folder
        create_tddft(mol_dir, found_log, mol_dir, func, basis, charge, multiplicity)

def copy_run_tddft():
    run = os.path.join(ROOT_DIR, RUN_SCRIPT_NAME)
    if not os.path.isfile(run):
        log("Error: run.sh missing for tddft copy")
        return
    for d in os.listdir(ROOT_DIR):
        td = os.path.join(d, "tddft")
        if os.path.isdir(td):
            run_target = os.path.join(td, RUN_SCRIPT_NAME)
            shutil.copy(run, run_target)
            com_files = [f for f in os.listdir(td) if f.endswith(".com")]
            if not com_files:
                log(f"[WARNING] No .com file in {td}")
                continue
            com_file = com_files[0]
            job_name = os.path.splitext(com_file)[0]
            with open(run_target, 'r') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith("#SBATCH --job-name="):
                    lines[i] = f"#SBATCH --job-name={job_name}\n"
                if "JobFile=" in line:
                    lines[i] = f"JobFile={job_name}\n"
            with open(run_target, 'w') as f:
                f.writelines(lines)
            log(f"Copied run.sh and updated job name to {job_name} in {td}")

def submit_one_tddft(jdir, user_jobs, submitted_jobs):
    logs = [f for f in os.listdir(jdir) if f.endswith(".log")]
    if logs:
        log_file = os.path.join(jdir, logs[0])
        if is_job_done(log_file):
            log(f"[SKIP] done: {jdir}")
            return
        if any(jdir in line for line in user_jobs):
            log(f"[WAIT] running: {jdir}")
            return
    run = os.path.join(jdir, RUN_SCRIPT_NAME)
    if os.path.isfile(run):
        result = subprocess.run(
            ["sbatch", "run.sh"],
            cwd=jdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        if result.returncode == 0:
            m = re.search(r"Submitted batch job (\d+)", result.stdout)
            if m:
                jobid = m.group(1)
                submitted_jobs[jdir] = jobid
                log(f"[SUBMIT] {jdir}: jobid={jobid}")
            else:
                log(f"[ERROR] Could not parse jobid: {result.stdout.strip()}")
        else:
            log(f"[ERROR] Failed: {result.stderr.strip()}")
    else:
        log(f"[MISSING] run.sh in {jdir}")

def submit_all_tddft():
    jdirs = [os.path.join(d, "tddft") for d in os.listdir(ROOT_DIR)
             if os.path.isdir(os.path.join(d, "tddft"))]
    queue = sorted(jdirs)
    submitted_jobs = {}
    while queue:
        cnt = get_job_count(USER)
        user_jobs = get_user_jobs(USER)
        if cnt < MAX_JOBS:
            jd = queue.pop(0)
            submit_one_tddft(jd, user_jobs, submitted_jobs)
            time.sleep(5)
        else:
            log(f"[WAIT] {cnt} jobs in queue. Sleeping 60s...")
            time.sleep(60)
    log("[DONE] All TDDFT jobs submitted or skipped.")
    return submitted_jobs

def wait_for_tddft_completion(submitted_jobs, max_wait_minutes=5000, check_interval=120):
    log("[INFO] Waiting for all TDDFT jobs to complete...")
    start_time = time.time()
    still_running = dict(submitted_jobs)
    while still_running:
        finished = []
        for jd, jobid in still_running.items():
            if is_running_jobid(jobid):
                continue
            log_files = [f for f in os.listdir(jd) if f.endswith('.log')]
            if not log_files:
                log(f"[ERROR] {jd}: no log file")
                finished.append(jd)
                continue
            done = False
            for lf in log_files:
                with open(os.path.join(jd, lf), 'r', errors='ignore') as f:
                    if "Normal termination" in f.read():
                        done = True
                        break
            if done:
                log(f"[OK] {jd}: Normal termination")
            else:
                log(f"[ERROR] {jd}: finished without Normal termination")
            finished.append(jd)
        for jd in finished:
            still_running.pop(jd, None)
        if not still_running:
            log("[OK] All TDDFT jobs completed.")
            return
        if (time.time() - start_time) > max_wait_minutes * 60:
            log("[WARNING] Timeout reached. Still unfinished:")
            for jd in still_running:
                log(f"  - {jd} (jobid {still_running[jd]})")
            return
        log(f"[WAIT] {len(still_running)} TDDFT jobs still running. Sleeping {check_interval//60} min...")
        time.sleep(check_interval)
def extract_ev_data():
    def log(msg):
        print(msg)

    log("[INFO] Extracting HOMO/LUMO and first S1/T1 excitations...")

    mol_data = {}

    # HOMO/LUMO label
    def get_transition_label(f, t, total_filled):
        f, t = int(f), int(t)
        homo = total_filled
        def label(o):
            d = o - homo
            if d == 0: return "HOMO"
            elif d < 0: return f"HOMO{d}"
            elif d == 1: return "LUMO"
            else: return f"LUMO+{d-1}"
        return f"{label(f)} → {label(t)}"

    for root, _, files in os.walk('.'):
        for file_name in files:
            if not file_name.endswith('.log') or not file_name.startswith('opt-'):
                continue
            file_path = os.path.join(root, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                if not any("Normal termination" in l for l in lines):
                    log(f"[SKIP] {file_path} did not terminate normally.")
                    continue

                # Extract HOMO/LUMO
                occ_evals, virt_evals = [], []
                for l in lines:
                    if "occ. eigenvalues" in l:
                        occ_evals.extend([float(x) for x in l.split('--')[1].split()])
                    elif "virt. eigenvalues" in l:
                        virt_evals.extend([float(x) for x in l.split('--')[1].split()])
                total_filled = len(occ_evals)
                homo_energy = occ_evals[-1] if occ_evals else ''
                lumo_energy = virt_evals[0] if virt_evals else ''

                # Extract first Triplet and Singlet
                singlet = triplet = None
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if line.startswith("Excited State"):
                        parts = line.split()
                        state_num = int(parts[2].strip(':'))
                        multiplicity = parts[3]  # 'Triplet-A' or 'Singlet-A'
                        energy_ev = float(parts[4])
                        # Collect transitions (next indented lines)
                        transitions = []
                        j = i + 1
                        while j < len(lines) and re.match(r'\s*\d+\s*->\s*\d+\s+[-+]?\d*\.\d+', lines[j]):
                            m = re.match(r'\s*(\d+)\s*->\s*(\d+)\s+([-+]?\d*\.\d+)', lines[j])
                            if m:
                                transitions.append((m.group(1), m.group(2), m.group(3)))
                            j += 1
                        i = j - 1
                        if transitions:
                            # pick largest coefficient
                            max_trans = max(transitions, key=lambda x: abs(float(x[2])))
                            trans_label = get_transition_label(max_trans[0], max_trans[1], total_filled)
                            trans_label_str = f"{trans_label} ({max_trans[2]})"
                            entry = {"Energy": energy_ev, "Transition": trans_label_str}
                            if "Singlet" in multiplicity and singlet is None:
                                singlet = entry
                            elif "Triplet" in multiplicity and triplet is None:
                                triplet = entry
                    i += 1

                mol_name = os.path.splitext(file_name)[0].replace("opt-", "")
                mol_data[mol_name] = {
                    "Filename": mol_name,
                    "S1energy": singlet["Energy"] if singlet else "",
                    "T1energy": triplet["Energy"] if triplet else "",
                    "Homoenergy": homo_energy,
                    "Lumoenergy": lumo_energy,
                    "Transition_Triplet": triplet["Transition"] if triplet else "",
                    "Transition_Singlet": singlet["Transition"] if singlet else "",
                    "TotalFilledOrbitals": total_filled
                }

            except Exception as e:
                log(f"[ERROR] {file_path}: {e}")

    # Write to text file
    with open("Td_ev_Values.txt", "w", encoding="utf-8") as txtfile:
        headers = ['Filename', 'S1energy', 'T1energy', 'Homoenergy', 'Lumoenergy',
                   'Transition_Triplet', 'Transition_Singlet', 'TotalFilledOrbitals']
        widths = [20, 10, 10, 12, 12, 40, 40, 15]
        for h, w in zip(headers, widths):
            txtfile.write(h.ljust(w))
        txtfile.write("\n" + "-"*sum(widths) + "\n")
        for row in mol_data.values():
            for key, width in zip(headers, widths):
                value = str(row.get(key, ""))
                txtfile.write(value.ljust(width))
            txtfile.write("\n")

    log("[DONE] Td_ev_Values.txt generated successfully.")


if __name__ == "__main__":
    # Step 1: Choose input type-----------
    while True:
        input_type = input("Input type? Enter 'gjf' or 'smiles': ").strip().lower()
        if input_type in ['gjf', 'smiles']:
            break
        print("[ERROR] Please enter either 'gjf' or 'smiles'.")

    # Step 2: Gather settings-------------
    do_xtb, charge, multiplicity, func, basis = get_user_inputs()

    # Step 3: Choose workflow mode*****************
    run_mode = get_run_mode()

    # Step 4: Process SMILES or GJF**********************
    if input_type == "smiles":
        input_file = "smiles.txt"
        if not os.path.isfile(input_file):
            log(f"[ERROR] SMILES input file '{input_file}' not found.")
            exit(1)

        with open(input_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    smiles_str, fname = line.split()
                    convert_smiles_to_gjf(smiles_str, fname, charge, multiplicity)
                    gjf_path = fname + ".gjf"
                    if do_xtb:
                        run_xtb_optimization_with_crest(gjf_path, charge, multiplicity)
                except ValueError:
                    print(f"[ERROR] Invalid line in input: '{line}'")
    else:
        log("[INFO] Proceeding with existing .gjf files in the directory.")
        if do_xtb:
            for f in os.listdir('.'):
                if f.endswith('.gjf'):
                    log(f"[INFO] Running xTB optimization on {f}")
                    run_xtb_optimization_with_crest(f, charge, multiplicity)

    # Step 5: Run Optimization****************************************
    mult_map = {1: "singlet", 2: "doublet", 3: "triplet"}
    mode_name = mult_map.get(multiplicity, f"mult{multiplicity}")

    prepare_com_files(func, basis, charge, multiplicity, mode_name)
    copy_run_script(mode_name)
    update_jobnames()

    submitted_opt = submit_optimization_jobs(mode_name)
    wait_for_optimization_completion(submitted_opt, mode_name)

    # Final negative frequency check********************************************
    bads = scan_logs()
    if bads:
        log("[WARNING] Negative frequencies found:")
        for path, freq in bads.items():
            print(f"  - {path}: {freq} cm^-1")
    else:
        log("[OK] No negative frequencies detected.")

    # Step 6: If user selected TDDFT, run TDDFT********************************
    if run_mode == 2:
        log("[INFO] Generating TDDFT inputs in top-level tddft folders...")
        gen_tddft(bads, func, basis, charge, multiplicity)

        log("[INFO] Copying run.sh into tddft folders...")
        copy_run_tddft()

        log("[INFO] Submitting all TDDFT jobs...")
        submitted_tddft = submit_all_tddft()

        log("[INFO] Waiting for all TDDFT jobs to complete...")
        wait_for_tddft_completion(submitted_tddft)

        log("[INFO] Extracting excited-state data...")
        extract_ev_data()

