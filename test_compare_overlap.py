import numpy as np
from gbasis.parsers import parse_gbs
from gbasis.wrappers import make_contractions
from gbasis.integrals.overlap import Overlap
from gbasis.integrals.overlap_n import contracted_n_overlap

atoms = ["H", "H"]
coords = np.array([[0.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0]])

basis_dict = parse_gbs("tests/data_631g.gbs")
basis = make_contractions(basis_dict, atoms, coords, coord_types="c")

print("Number of shells:", len(basis))

# Reference overlap (NO screening)
olp_ref = Overlap.construct_array(basis)


print("Reference overlap shape:", olp_ref.shape)

n_shells = len(basis)

total_ao = olp_ref.shape[0]
olp_mine = np.zeros((total_ao, total_ao))

ao_offset_i = 0

for i, shell_i in enumerate(basis):

    Li = shell_i.num_cart
    Mi = shell_i.num_seg_cont
    size_i = Li * Mi

    ao_offset_j = 0

    for j, shell_j in enumerate(basis):

        Lj = shell_j.num_cart
        Mj = shell_j.num_seg_cont
        size_j = Lj * Mj

        # Compute N=2 overlap
        block = contracted_n_overlap([shell_i, shell_j])

        block = block.reshape(size_i, size_j)

        olp_mine[
            ao_offset_i:ao_offset_i+size_i,
            ao_offset_j:ao_offset_j+size_j
        ] = block

        ao_offset_j += size_j

    ao_offset_i += size_i


print("My overlap shape:", olp_mine.shape)
# Compare
diff = np.max(np.abs(olp_ref - olp_mine))

print("Maximum absolute difference:", diff)
print("All close (1e-10)?", np.allclose(olp_ref, olp_mine, atol=1e-10))

