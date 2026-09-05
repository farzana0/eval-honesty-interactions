"""Plant known principal angles between two subspaces and recover them."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from direction_utils import principal_angles


def main() -> int:
    rng = np.random.default_rng(0)
    d, k = 64, 3
    ok = True
    # Build an orthonormal frame, then rotate B's columns away from A's by
    # planted angles theta_i inside 2D planes spanned by (a_i, u_i) with u_i
    # orthogonal to all of A.
    Q, _ = np.linalg.qr(rng.normal(size=(d, 2 * k)))
    A = Q[:, :k]
    U = Q[:, k:2 * k]
    planted = np.array([0.1, 0.7, 1.3])  # radians, ascending
    B = A * np.cos(planted) + U * np.sin(planted)

    got = principal_angles(A, B)
    err = np.abs(np.sort(got) - np.sort(planted)).max()
    print(f"planted={planted}  recovered={np.round(np.sort(got), 6)}  max err={err:.2e}")
    ok &= err < 1e-9

    # identical subspaces -> all angles 0
    z = principal_angles(A, A @ rng.normal(size=(k, k)))  # any basis of the same span
    print(f"same-span angles: {np.round(z, 9)}")
    ok &= np.abs(z).max() < 1e-9

    # orthogonal subspaces -> all angles pi/2
    o = principal_angles(A, U)
    print(f"orthogonal-span angles: {np.round(o, 9)}")
    ok &= np.abs(o - np.pi / 2).max() < 1e-9

    # single vectors: angle equals arccos of the cosine
    a = rng.normal(size=d); b = rng.normal(size=d)
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    pa = principal_angles(a[:, None], b[:, None])[0]
    print(f"1-d: arccos(cos)={np.arccos(abs(c)):.9f}  principal={pa:.9f}")
    ok &= abs(pa - np.arccos(abs(c))) < 1e-9

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
