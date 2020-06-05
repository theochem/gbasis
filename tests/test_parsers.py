"""Test gbasis.parsers."""
from gbasis.parsers import make_contractions, parse_gbs, parse_nwchem
import numpy as np
import pytest
from utils import find_datafile


def test_parse_nwchem_sto6g():
    """Test gbasis.parsers.parse_nwchem for sto6g."""
    test = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    # test specific cases
    assert len(test["H"]) == 1
    assert test["H"][0][0] == 0
    assert np.allclose(
        test["H"][0][1],
        np.array([35.52322122, 6.513143725, 1.822142904, 0.625955266, 0.243076747, 0.100112428]),
    )
    assert np.allclose(
        test["H"][0][2],
        np.array(
            [
                0.00916359628,
                0.04936149294,
                0.16853830490,
                0.37056279970,
                0.41649152980,
                0.13033408410,
            ]
        ).reshape(6, 1),
    )

    assert len(test["Li"]) == 3
    assert test["Li"][0][0] == 0
    assert np.allclose(
        test["Li"][0][1],
        np.array(
            [167.17584620, 30.651508400, 8.5751874770, 2.9458083370, 1.1439435810, 0.4711391391]
        ),
    )
    assert np.allclose(
        test["Li"][0][2],
        np.array(
            [
                0.00916359628,
                0.04936149294,
                0.16853830490,
                0.37056279970,
                0.41649152980,
                0.13033408410,
            ]
        ).reshape(6, 1),
    )
    assert test["Li"][1][0] == 0
    assert np.allclose(
        test["Li"][1][1],
        np.array(
            [6.5975639810, 1.3058300920, 0.4058510193, 0.1561455158, 0.0678141039, 0.0310841655]
        ),
    )
    assert np.allclose(
        test["Li"][1][2],
        np.array(
            [
                -0.01325278809,
                -0.04699171014,
                -0.03378537151,
                0.25024178610,
                0.59511725260,
                0.24070617630,
            ]
        ),
    )
    assert test["Li"][2][0] == 1
    assert np.allclose(
        test["Li"][2][1],
        np.array(
            [6.5975639810, 1.3058300920, 0.4058510193, 0.1561455158, 0.0678141039, 0.0310841655]
        ),
    )
    assert np.allclose(
        test["Li"][2][2],
        np.array(
            [0.0037596966, 0.0376793698, 0.1738967435, 0.4180364347, 0.4258595477, 0.1017082955]
        ),
    )

    assert len(test["Kr"]) == 8
    assert test["Kr"][0][0] == 0
    assert np.allclose(
        test["Kr"][0][1],
        np.array(
            [
                0.2885373644e05,
                0.5290300991e04,
                0.1480035573e04,
                0.5084321647e03,
                0.1974390878e03,
                0.8131631964e02,
            ]
        ),
    )
    assert np.allclose(
        test["Kr"][0][2],
        np.array(
            [
                0.9163596280e-02,
                0.4936149294e-01,
                0.1685383049e00,
                0.3705627997e00,
                0.4164915298e00,
                0.1303340841e00,
            ]
        ).reshape(6, 1),
    )
    assert test["Kr"][3][0] == 0
    assert np.allclose(
        test["Kr"][3][1],
        np.array(
            [
                0.1284965766e03,
                0.3700047197e02,
                0.1455321200e02,
                0.6651312517e01,
                0.3310132155e01,
                0.1699958288e01,
            ]
        ),
    )
    assert np.allclose(
        test["Kr"][3][2],
        np.array(
            [
                -0.7943126362e-02,
                -0.7100264172e-01,
                -0.1785026925e00,
                0.1510635058e00,
                0.7354914767e00,
                0.2760593123e00,
            ]
        ),
    )
    assert test["Kr"][4][0] == 1
    assert np.allclose(
        test["Kr"][4][1],
        np.array(
            [
                0.1284965766e03,
                0.3700047197e02,
                0.1455321200e02,
                0.6651312517e01,
                0.3310132155e01,
                0.1699958288e01,
            ]
        ),
    )
    assert np.allclose(
        test["Kr"][4][2],
        np.array(
            [
                -0.7139358907e-02,
                -0.1829277070e-01,
                0.7621621428e-01,
                0.4145098597e00,
                0.4889621471e00,
                0.1058816521e00,
            ]
        ),
    )
    assert test["Kr"][7][0] == 2
    assert np.allclose(
        test["Kr"][7][1],
        np.array(
            [
                0.1284965766e03,
                0.3700047197e02,
                0.1455321200e02,
                0.6651312517e01,
                0.3310132155e01,
                0.1699958288e01,
            ]
        ),
    )
    assert np.allclose(
        test["Kr"][7][2],
        np.array(
            [
                0.6633434386e-02,
                0.5958177963e-01,
                0.2401949582e00,
                0.4648114679e00,
                0.3434092326e00,
                0.5389056980e-01,
            ]
        ).reshape(6, 1),
    )


def test_parse_nwchem_anorcc():
    """Test gbasis.parsers.parse_nwchem for anorcc."""
    test = parse_nwchem(find_datafile("data_anorcc.nwchem"))
    # test specific cases
    assert len(test["H"]) == 4
    assert test["H"][0][0] == 0
    assert np.allclose(
        test["H"][0][1],
        np.array(
            [
                188.6144500000,
                28.2765960000,
                6.4248300000,
                1.8150410000,
                0.5910630000,
                0.2121490000,
                0.0798910000,
                0.0279620000,
            ]
        ),
    )
    assert np.allclose(
        test["H"][0][2],
        np.array(
            [
                [0.00096385, -0.0013119, 0.00242240, -0.0115701, 0.01478099, -0.0212892],
                [0.00749196, -0.0103451, 0.02033817, -0.0837154, 0.09403187, -0.1095596],
                [0.03759541, -0.0504953, 0.08963935, -0.4451663, 0.53618016, -1.4818260],
                [0.14339498, -0.2073855, 0.44229071, -1.1462710, -0.6089639, 3.0272963],
                [0.34863630, -0.4350885, 0.57571439, 2.5031871, -1.1148890, -3.7630860],
                [0.43829736, -0.0247297, -0.9802890, -1.5828490, 3.4820812, 3.6574131],
                [0.16510661, 0.32252599, -0.6721538, 0.03096569, -3.7625390, -2.5012370],
                [0.02102287, 0.70727538, 1.1417685, 0.30862864, 1.6766932, 0.89405394],
            ]
        ),
    )

    assert test["H"][1][0] == 1
    assert np.allclose(
        test["H"][1][1], np.array([2.3050000000, 0.8067500000, 0.2823620000, 0.0988270000])
    )
    assert np.allclose(
        test["H"][1][2],
        np.array(
            [
                [0.11279019, -0.2108688, 0.75995011, -1.4427420],
                [0.41850753, -0.5943796, 0.16461590, 2.3489914],
                [0.47000773, 0.08968888, -1.3710140, -1.9911520],
                [0.18262603, 0.86116340, 1.0593155, 0.90505601],
            ]
        ),
    )

    assert test["H"][2][0] == 2
    assert np.allclose(test["H"][2][1], np.array([1.8190000000, 0.7276000000, 0.2910400000]))
    assert np.allclose(
        test["H"][2][2],
        np.array(
            [
                [0.27051341, -0.7938035, 1.3082770],
                [0.55101250, -0.0914252, -2.0210590],
                [0.33108664, 0.86200334, 1.2498888],
            ]
        ),
    )

    assert test["H"][3][0] == 3
    assert np.allclose(test["H"][3][1], np.array([0.9701090000]))
    assert np.allclose(test["H"][3][2], np.array([[1.0000000]]))


def test_parse_gbs_sto6g():
    """Test gbasis.parsers.parse_gbs for sto6g."""
    test = parse_gbs(find_datafile("data_sto6g.gbs"))
    # test specific cases
    assert len(test["H"]) == 1
    assert test["H"][0][0] == 0
    assert np.allclose(
        test["H"][0][1],
        np.array([35.52322122, 6.513143725, 1.822142904, 0.625955266, 0.243076747, 0.100112428]),
    )
    assert np.allclose(
        test["H"][0][2],
        np.array(
            [
                0.00916359628,
                0.04936149294,
                0.16853830490,
                0.37056279970,
                0.41649152980,
                0.13033408410,
            ]
        ).reshape(6, 1),
    )

    assert len(test["Li"]) == 3
    assert test["Li"][0][0] == 0
    assert np.allclose(
        test["Li"][0][1],
        np.array(
            [167.17584620, 30.651508400, 8.5751874770, 2.9458083370, 1.1439435810, 0.4711391391]
        ),
    )
    assert np.allclose(
        test["Li"][0][2],
        np.array(
            [
                0.00916359628,
                0.04936149294,
                0.16853830490,
                0.37056279970,
                0.41649152980,
                0.13033408410,
            ]
        ).reshape(6, 1),
    )
    assert test["Li"][1][0] == 0
    assert np.allclose(
        test["Li"][1][1],
        np.array(
            [6.5975639810, 1.3058300920, 0.4058510193, 0.1561455158, 0.0678141039, 0.0310841655]
        ),
    )
    assert np.allclose(
        test["Li"][1][2],
        np.array(
            [
                -0.01325278809,
                -0.04699171014,
                -0.03378537151,
                0.25024178610,
                0.59511725260,
                0.24070617630,
            ]
        ).reshape(-1, 1),
    )
    assert test["Li"][2][0] == 1
    assert np.allclose(
        test["Li"][2][1],
        np.array(
            [6.5975639810, 1.3058300920, 0.4058510193, 0.1561455158, 0.0678141039, 0.0310841655]
        ),
    )
    assert np.allclose(
        test["Li"][2][2],
        np.array(
            [0.0037596966, 0.0376793698, 0.1738967435, 0.4180364347, 0.4258595477, 0.1017082955]
        ).reshape(-1, 1),
    )

    assert len(test["Kr"]) == 8
    assert test["Kr"][0][0] == 0
    assert np.allclose(
        test["Kr"][0][1],
        np.array(
            [
                0.2885373644e05,
                0.5290300991e04,
                0.1480035573e04,
                0.5084321647e03,
                0.1974390878e03,
                0.8131631964e02,
            ]
        ),
    )
    assert np.allclose(
        test["Kr"][0][2],
        np.array(
            [
                0.9163596280e-02,
                0.4936149294e-01,
                0.1685383049e00,
                0.3705627997e00,
                0.4164915298e00,
                0.1303340841e00,
            ]
        ).reshape(6, 1),
    )
    assert test["Kr"][5][0] == 0
    assert np.allclose(
        test["Kr"][5][1],
        np.array(
            [
                0.1284965766e03,
                0.3700047197e02,
                0.1455321200e02,
                0.6651312517e01,
                0.3310132155e01,
                0.1699958288e01,
            ]
        ),
    )
    assert np.allclose(
        test["Kr"][5][2],
        np.array(
            [
                -0.9737395526e-02,
                -0.7265876782e-01,
                -0.1716155198e00,
                0.1289776243e00,
                0.7288614510e00,
                0.3013317422e00,
            ]
        ).reshape(-1, 1),
    )
    assert test["Kr"][6][0] == 1
    assert np.allclose(
        test["Kr"][6][1],
        np.array(
            [
                0.1284965766e03,
                0.3700047197e02,
                0.1455321200e02,
                0.6651312517e01,
                0.3310132155e01,
                0.1699958288e01,
            ]
        ),
    )
    assert np.allclose(
        test["Kr"][6][2],
        np.array(
            [
                -0.8104943356e-02,
                -0.1715478915e-01,
                0.7369785762e-01,
                0.3965149986e00,
                0.4978084880e00,
                0.1174825823e00,
            ]
        ).reshape(-1, 1),
    )
    assert test["Kr"][7][0] == 2
    assert np.allclose(
        test["Kr"][7][1],
        np.array(
            [
                0.1284965766e03,
                0.3700047197e02,
                0.1455321200e02,
                0.6651312517e01,
                0.3310132155e01,
                0.1699958288e01,
            ]
        ),
    )
    assert np.allclose(
        test["Kr"][7][2],
        np.array(
            [
                0.6633434386e-02,
                0.5958177963e-01,
                0.2401949582e00,
                0.4648114679e00,
                0.3434092326e00,
                0.5389056980e-01,
            ]
        ).reshape(6, 1),
    )


def test_parse_gbs_631g():
    """Test gbasis.parsers.parse_gbs for 6-31G."""
    test = parse_gbs(find_datafile("data_631g.gbs"))
    # test specific cases
    assert len(test["H"]) == 2
    assert test["H"][0][0] == 0
    assert test["H"][1][0] == 0
    assert np.allclose(test["H"][0][1], np.array([18.73113696, 2.825394365, 0.6401216923]))
    assert np.allclose(
        test["H"][0][2], np.array([0.03349460434, 0.2347269535, 0.8137573261]).reshape(3, 1)
    )
    assert len(test["Li"]) == 5
    assert test["Li"][0][0] == 0
    assert np.allclose(
        test["Li"][0][1],
        np.array([642.4189150, 96.79851530, 22.09112120, 6.201070250, 1.935117680, 0.636735789]),
    )
    assert np.allclose(
        test["Li"][0][2],
        np.array(
            [
                0.002142607810,
                0.01620887150,
                0.07731557250,
                0.2457860520,
                0.4701890040,
                0.3454708450,
            ]
        ).reshape(6, 1),
    )
    assert test["Li"][1][0] == 0
    assert np.allclose(test["Li"][1][1], np.array([2.324918408, 0.6324303556, 0.07905343475]))
    assert np.allclose(
        test["Li"][1][2], np.array([-0.03509174574, -0.1912328431, 1.083987795]).reshape(-1, 1)
    )
    assert test["Li"][2][0] == 1
    assert np.allclose(test["Li"][2][1], np.array([2.324918408, 0.6324303556, 0.07905343475]))
    assert np.allclose(
        test["Li"][2][2], np.array([0.008941508043, 0.1410094640, 0.9453636953]).reshape(-1, 1)
    )

    assert test["Li"][3][0] == 0
    assert np.allclose(test["Li"][3][1], np.array([0.3596197175e-01]))
    assert np.allclose(test["Li"][3][2], np.array([0.100000000e01]))
    assert len(test["Kr"]) == 11
    assert test["Kr"][0][0] == 0
    assert np.allclose(
        test["Kr"][0][1],
        np.array(
            [
                0.1205524000e06,
                0.1810225000e05,
                0.4124126000e04,
                0.1163472000e04,
                0.3734612000e03,
                0.1280897000e03,
            ]
        ),
    )
    assert np.allclose(
        test["Kr"][0][2],
        np.array(
            [
                0.1714050000e-02,
                0.1313805000e-01,
                0.6490006000e-01,
                0.2265185000e00,
                0.4764961000e00,
                0.3591952000e00,
            ]
        ).reshape(6, 1),
    )
    assert test["Kr"][4][0] == 1
    assert np.allclose(
        test["Kr"][4][1],
        np.array(
            [
                0.1175107000e03,
                0.4152553000e02,
                0.1765290000e02,
                0.7818313000e01,
                0.3571775000e01,
                0.1623750000e01,
            ]
        ),
    )
    assert np.allclose(
        test["Kr"][4][2],
        np.array(
            [
                -0.6922855000e-02,
                -0.3069239000e-01,
                0.4480260000e-01,
                0.3636775000e00,
                0.4952412000e00,
                0.2086340000e00,
            ]
        ).reshape(-1, 1),
    )

    assert test["Kr"][5][0] == 0
    assert np.allclose(test["Kr"][5][1], np.array([0.2374560e01, 0.8691930e00, 0.3474730e00]))
    assert np.allclose(
        test["Kr"][5][2],
        np.array([0.3251184000e00, -0.2141533000e00, -0.9755083000e00]).reshape(-1, 1),
    )
    assert test["Kr"][8][0] == 1
    assert np.allclose(test["Kr"][8][1], np.array([0.1264790000e00]))
    assert np.allclose(test["Kr"][8][2], np.array([0.1e01]))
    assert test["Kr"][9][0] == 2
    assert np.allclose(test["Kr"][9][1], np.array([0.6853888e02, 0.1914333e02, 0.6251213e01]))
    assert np.allclose(
        test["Kr"][9][2], np.array([0.7530705e-01, 0.3673551e00, 0.7120146e00]).reshape(3, 1)
    )
    assert test["Kr"][10][0] == 2
    assert np.allclose(test["Kr"][10][1], np.array([0.1979236000e01]))
    assert np.allclose(test["Kr"][10][2], np.array([0.1000000000e01]))


def test_parse_nwchem_631g():
    """Test gbasis.parsers.parse_nwchem for 6-31G."""
    test = parse_nwchem(find_datafile("data_631g.nwchem"))
    # test specific cases
    assert len(test["H"]) == 2
    assert test["H"][0][0] == 0
    assert test["H"][1][0] == 0
    assert np.allclose(test["H"][0][1], np.array([18.73113696, 2.825394365, 0.6401216923]))
    assert np.allclose(
        test["H"][0][2], np.array([0.03349460434, 0.2347269535, 0.8137573261]).reshape(3, 1)
    )
    assert len(test["Li"]) == 5
    assert test["Li"][0][0] == 0
    assert np.allclose(
        test["Li"][0][1],
        np.array([642.4189150, 96.79851530, 22.09112120, 6.201070250, 1.935117680, 0.636735789]),
    )
    assert np.allclose(
        test["Li"][0][2],
        np.array(
            [
                0.002142607810,
                0.01620887150,
                0.07731557250,
                0.2457860520,
                0.4701890040,
                0.3454708450,
            ]
        ).reshape(6, 1),
    )
    assert test["Li"][1][0] == 0
    assert np.allclose(test["Li"][1][1], np.array([2.324918408, 0.6324303556, 0.07905343475]))
    assert np.allclose(test["Li"][1][2], np.array([-0.03509174574, -0.1912328431, 1.083987795]))
    assert test["Li"][2][0] == 1
    assert np.allclose(test["Li"][2][1], np.array([2.324918408, 0.6324303556, 0.07905343475]))
    assert np.allclose(test["Li"][2][2], np.array([0.008941508043, 0.1410094640, 0.9453636953]))

    assert test["Li"][3][0] == 0
    assert np.allclose(test["Li"][3][1], np.array([0.3596197175e-01]))
    assert np.allclose(test["Li"][3][2], np.array([0.100000000e01]).reshape(-1, 1))
    assert len(test["Kr"]) == 11
    assert test["Kr"][0][0] == 0
    assert np.allclose(
        test["Kr"][0][1],
        np.array(
            [
                0.1205524000e06,
                0.1810225000e05,
                0.4124126000e04,
                0.1163472000e04,
                0.3734612000e03,
                0.1280897000e03,
            ]
        ),
    )
    assert np.allclose(
        test["Kr"][0][2],
        np.array(
            [
                0.1714050000e-02,
                0.1313805000e-01,
                0.6490006000e-01,
                0.2265185000e00,
                0.4764961000e00,
                0.3591952000e00,
            ]
        ).reshape(6, 1),
    )
    assert test["Kr"][4][0] == 1
    assert np.allclose(
        test["Kr"][4][1],
        np.array(
            [
                0.1175107000e03,
                0.4152553000e02,
                0.1765290000e02,
                0.7818313000e01,
                0.3571775000e01,
                0.1623750000e01,
            ]
        ),
    )
    assert np.allclose(
        test["Kr"][4][2],
        np.array(
            [
                -0.6922855000e-02,
                -0.3069239000e-01,
                0.4480260000e-01,
                0.3636775000e00,
                0.4952412000e00,
                0.2086340000e00,
            ]
        ),
    )

    assert test["Kr"][5][0] == 0
    assert np.allclose(test["Kr"][5][1], np.array([0.23745600e01, 0.8691930e00, 0.34747300e00]))
    assert np.allclose(test["Kr"][5][2], np.array([0.3251184e00, -0.2141533e00, -0.9755083e00]))
    assert test["Kr"][8][0] == 1
    assert np.allclose(test["Kr"][8][1], np.array([0.1264790000e00]))
    assert np.allclose(test["Kr"][8][2], np.array([0.1e01]).reshape(1, 1))
    assert test["Kr"][9][0] == 2
    assert np.allclose(test["Kr"][9][1], np.array([0.6853888e02, 0.1914333e02, 0.6251213e01]))
    assert np.allclose(
        test["Kr"][9][2], np.array([0.7530705e-01, 0.3673551e00, 0.7120146e00]).reshape(3, 1)
    )
    assert test["Kr"][10][0] == 2
    assert np.allclose(test["Kr"][10][1], np.array([0.197923600e01]))
    assert np.allclose(test["Kr"][10][2], np.array([0.10000000e01]))


def test_parse_gbs_anorcc():
    """Test gbasis.parsers.parse_gbs for anorcc."""
    test = parse_gbs(find_datafile("data_anorcc.gbs"))
    # test specific cases
    assert len(test["H"]) == 4
    assert test["H"][0][0] == 0
    assert np.allclose(
        test["H"][0][1],
        np.array(
            [
                188.6144500000,
                28.2765960000,
                6.4248300000,
                1.8150410000,
                0.5910630000,
                0.2121490000,
                0.0798910000,
                0.0279620000,
            ]
        ),
    )
    assert np.allclose(
        test["H"][0][2],
        np.array(
            [
                [0.00096385, -0.0013119, 0.00242240, -0.0115701, 0.01478099, -0.0212892],
                [0.00749196, -0.0103451, 0.02033817, -0.0837154, 0.09403187, -0.1095596],
                [0.03759541, -0.0504953, 0.08963935, -0.4451663, 0.53618016, -1.4818260],
                [0.14339498, -0.2073855, 0.44229071, -1.1462710, -0.6089639, 3.0272963],
                [0.34863630, -0.4350885, 0.57571439, 2.5031871, -1.1148890, -3.7630860],
                [0.43829736, -0.0247297, -0.9802890, -1.5828490, 3.4820812, 3.6574131],
                [0.16510661, 0.32252599, -0.6721538, 0.03096569, -3.7625390, -2.5012370],
                [0.02102287, 0.70727538, 1.1417685, 0.30862864, 1.6766932, 0.89405394],
            ]
        ),
    )

    assert test["H"][1][0] == 1
    assert np.allclose(
        test["H"][1][1], np.array([2.3050000000, 0.8067500000, 0.2823620000, 0.0988270000])
    )
    assert np.allclose(
        test["H"][1][2],
        np.array(
            [
                [0.11279019, -0.2108688, 0.75995011, -1.4427420],
                [0.41850753, -0.5943796, 0.16461590, 2.3489914],
                [0.47000773, 0.08968888, -1.3710140, -1.9911520],
                [0.18262603, 0.86116340, 1.0593155, 0.90505601],
            ]
        ),
    )

    assert test["H"][2][0] == 2
    assert np.allclose(test["H"][2][1], np.array([1.8190000000, 0.7276000000, 0.2910400000]))
    assert np.allclose(
        test["H"][2][2],
        np.array(
            [
                [0.27051341, -0.7938035, 1.3082770],
                [0.55101250, -0.0914252, -2.0210590],
                [0.33108664, 0.86200334, 1.2498888],
            ]
        ),
    )

    assert test["H"][3][0] == 3
    assert np.allclose(test["H"][3][1], np.array([0.9701090000]))
    assert np.allclose(test["H"][3][2], np.array([[1.0000000]]))


def test_make_contractions():
    """Test gbasis.contractions.make_contractions."""
    basis_dict = parse_nwchem(find_datafile("data_sto6g.nwchem"))
    with pytest.raises(TypeError):
        make_contractions(basis_dict, {"H", "H"}, np.array([[0, 0, 0], [1, 1, 1]]))
    with pytest.raises(TypeError):
        make_contractions(basis_dict, [0, 0], np.array([[0, 0, 0], [1, 1, 1]]))

    with pytest.raises(TypeError):
        make_contractions(basis_dict, ["H", "H"], [[0, 0, 0], [1, 1, 1]])
    with pytest.raises(TypeError):
        make_contractions(basis_dict, ["H", "H"], np.array([0, 0, 0, 1, 1, 1]))
    with pytest.raises(TypeError):
        make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0, 2], [1, 1, 1, 2]]))

    with pytest.raises(ValueError):
        make_contractions(basis_dict, ["H", "H", "H"], np.array([[0, 0, 0], [1, 1, 1]]))

    with pytest.raises(TypeError):
        make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]), [0, 0])
    with pytest.raises(TypeError):
        make_contractions(
            basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]), [0, 0], overlap=False
        )

    test = make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]))
    assert isinstance(test, tuple)
    assert len(test) == 2
    assert test[0].angmom == 0
    assert np.allclose(test[0].coord, np.array([0, 0, 0]))
    assert np.allclose(
        test[0].coeffs,
        np.array(
            [
                0.00916359628,
                0.04936149294,
                0.16853830490,
                0.37056279970,
                0.41649152980,
                0.13033408410,
            ]
        ).reshape(6, 1),
    )
    assert np.allclose(
        test[0].exps,
        np.array([35.52322122, 6.513143725, 1.822142904, 0.625955266, 0.243076747, 0.100112428]),
    )
    assert test[1].angmom == 0
    assert np.allclose(test[1].coord, np.array([1, 1, 1]))
    assert np.allclose(
        test[1].coeffs,
        np.array(
            [
                0.00916359628,
                0.04936149294,
                0.16853830490,
                0.37056279970,
                0.41649152980,
                0.13033408410,
            ]
        ).reshape(6, 1),
    )
    assert np.allclose(
        test[1].exps,
        np.array([35.52322122, 6.513143725, 1.822142904, 0.625955266, 0.243076747, 0.100112428]),
    )


def test_make_contractions_gbs():
    """Test gbasis.contractions.make_contractions."""
    basis_dict = parse_gbs(find_datafile("data_631g.gbs"))
    with pytest.raises(TypeError):
        make_contractions(basis_dict, {"H", "H"}, np.array([[0, 0, 0], [1, 1, 1]]))
    with pytest.raises(TypeError):
        make_contractions(basis_dict, [0, 0], np.array([[0, 0, 0], [1, 1, 1]]))

    with pytest.raises(TypeError):
        make_contractions(basis_dict, ["H", "H"], [[0, 0, 0], [1, 1, 1]])
    with pytest.raises(TypeError):
        make_contractions(basis_dict, ["H", "H"], np.array([0, 0, 0, 1, 1, 1]))
    with pytest.raises(TypeError):
        make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0, 2], [1, 1, 1, 2]]))

    with pytest.raises(ValueError):
        make_contractions(basis_dict, ["H", "H", "H"], np.array([[0, 0, 0], [1, 1, 1]]))

    with pytest.raises(TypeError):
        make_contractions(basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]), [0, 0])
    with pytest.raises(TypeError):
        make_contractions(
            basis_dict, ["H", "H"], np.array([[0, 0, 0], [1, 1, 1]]), [0, 0], overlap=False
        )
