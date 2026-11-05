import re
from pathlib import Path

path = Path("paper/outage_conditioned_prioritization/paper_topology_coverage_restoration_restructured.tex")
text = path.read_text(encoding="utf-8")

print("Total characters:", len(text))

# Let's inspect sections
sections = re.split(r"\\section\{", text)
for sec in sections[1:]:
    sec_title = sec.split("}")[0]
    print(f"\n=======================================================")
    print(f"=== Section: {sec_title} ===")
    print(f"=======================================================")
    
    # Check coverage symbols
    c_matches = re.findall(r"c_[a-zA-Z0-9,\\]+\([^\)]+\)", sec)
    if c_matches:
        print("Coverage notation c_d(A):", set(c_matches))
        
    u_matches = re.findall(r"U\([^\)]+\)", sec)
    if u_matches:
        print("Tail coverage notation U(A):", set(u_matches))
        
    delta_matches = re.findall(r"\\Delta[^\$]*", sec)
    if delta_matches:
        print("Delta notation:", set(delta_matches[:10]))
        
    q_matches = re.findall(r"q_\{?[a-zA-Z0-9,\\]+\}?", sec)
    if q_matches:
        print("Weight notation q:", set(q_matches))
        
    r_matches = re.findall(r"R_\{?[a-zA-Z0-9,\\]+\}?", sec)
    if r_matches:
        print("Response notation R:", set(r_matches))
        
    rho_matches = re.findall(r"\\rho[a-zA-Z0-9_,\{\}\\]*", sec)
    if rho_matches:
        print("Rho notation:", set(rho_matches))
