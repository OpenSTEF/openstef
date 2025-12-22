"""Regional mapping for Liander 2024 benchmark targets.

Maps each target to its official Dutch school vacation region (north, middle, south)
based on the government classification from:
https://www.rijksoverheid.nl/onderwerpen/schoolvakanties/regios-schoolvakantie

OFFICIAL REGIONS:
- Regio north: Drenthe, Flevoland (excl. Zeewolde), Friesland, Gelderland (only Hattem),
                Groningen, north-Holland, Overijssel, Utrecht (Eemnes + former Abcoude)
- Regio middle: Flevoland (Zeewolde), Gelderland (Apeldoorn, Barneveld, Ede, etc.),
                 north-Brabant (Altena excl. Hank/Dussen), Utrecht (excl. Eemnes/Abcoude),
                 south-Holland (all)
- Regio south: Gelderland (Arnhem, Nijmegen, etc.), Limburg, north-Brabant (most),
              Zeeland (all)
"""

from typing import Literal

DutchSchoolVacationRegion = Literal["north", "middle", "south"]

_TARGET_TO_REGION: dict[str, DutchSchoolVacationRegion] = {
    # REGIO north - Based on official government classification
    # Friesland (all municipalities)
    "Within 20 kilometers of Leeuwarden_normalized": "north",
    "Within 5 kilometers of Leeuwarden_normalized": "north",
    "SS Harlingen": "north",
    "OS Sneek": "north",
    "SS Ureterp": "north",
    "OS Gorredijk": "north",
    "SS Heerenveen Pim Mulier": "north",
    "RS Grouw (Grou)": "north",
    "RS Surhuisterveen": "north",
    "SS northwolde": "north",
    "OS Bergum": "north",
    "RS Waskemeer": "north",
    "OS Drachten": "north",
    "Within 15 kilometers of Oosterwolde_normalized": "north",
    # Groningen (all municipalities)
    "RS Hallum": "north",
    "SS Stiens": "north",
    "OS Herbayum": "north",
    # north-Holland (all municipalities)
    "RS Wieringerwerf": "north",
    "OS middlemeer": "north",
    "RS Minnertsga": "north",
    "OS Edam": "north",
    "Within 15 kilometers of Opmeer_normalized": "north",
    "OS Texel": "north",
    "OS Naarden": "north",  # north-Holland
    "OS Weesp": "north",  # north-Holland
    "OS Waarderpolder": "north",  # north-Holland (Haarlemmermeer)
    "OS Watergraafsmeer": "north",  # north-Holland (Amsterdam)
    "OS Amsterdam Hemweg": "north",  # north-Holland (Amsterdam)
    "Within 10 kilometers of Westwoud_normalized": "north",  # north-Holland
    # Overijssel (all municipalities)
    "Within 15 kilometers of Zutphen_normalized": "north",  # Overijssel border
    # Flevoland (all except Zeewolde)
    "Within 15 kilometers of Dronten_normalized": "north",
    "OS Lelystad": "north",
    "OS Almere": "north",  # Flevoland
    # REGIO middle - Based on official government classification
    # Gelderland (specific municipalities: Apeldoorn, Barneveld, Ede, Doetinchem, etc.)
    "OS Apeldoorn": "middle",  # Gelderland - middle
    "OS Doetinchem": "middle",  # Gelderland - middle
    "OS Ede": "middle",  # Gelderland - middle
    "OS Eibergen": "middle",  # Gelderland - Oost-Gelre - middle
    "RS Anklaar": "middle",  # Gelderland - Oude IJsselstreek - middle
    "OS Tiel": "middle",  # Gelderland - middle
    "RS Wamel": "middle",  # Gelderland - West Betuwe - middle
    # south-Holland (all municipalities)
    "Within 15 kilometers of Alphen aan den Rijn_normalized": "middle",
    "OS Leiden north": "middle",
    "OS Sassenheim": "middle",
    "OS Oterleek": "middle",  # south-Holland (Rotterdam area)
    "Within 20 kilometers of Rotterdam_normalized": "middle",
    "OS Westhaven": "middle",  # south-Holland (Rotterdam)
    "OS Zevenhuizen": "middle",  # south-Holland
    # REGIO south - Based on official government classification
    # Gelderland (Arnhem, Nijmegen, Zevenaar, etc.)
    "Within Stadsregio Arnhem Nijmegen_normalized": "south",
    "OS Nijmegen": "south",  # Gelderland - south
    "RS Poederoijen": "south",  # Gelderland - Buren - south
    "RS Roodwillingen (Duiven)": "south",  # Gelderland - Duiven - south
    "RS Angerlo": "south",  # Gelderland - Zevenaar/Montferland - south
    "OS Zevenaar": "south",  # Gelderland - south
}


class TargetGeoMapper:
    """Maps Liander 2024 benchmark targets to Dutch school vacation regions.

    Provides methods to map target names to their official school vacation regions
    based on geographical location in the Netherlands.

    Example:
        >>> mapper = TargetGeoMapper()
        >>> mapper.get_region("OS Apeldoorn")
        'middle'
        >>> mapper.get_region("OS Nijmegen")
        'south'
        >>> mapper.get_all_regions()
        {'north', 'middle', 'south'}
    """

    def __init__(self) -> None:
        """Initialize the regional mapper with target mappings."""
        self._mapping = _TARGET_TO_REGION
        self.DutchSchoolVacationRegion = Literal["north", "middle", "south"]

    def get_region(self, target_name: str) -> DutchSchoolVacationRegion:
        """Get the official Dutch school vacation region for a target name.

        Args:
            target_name: Name of the benchmark target

        Returns:
            Region code: 'north', 'middle', or 'south'
            Defaults to 'middle' if target not found in mapping
        """
        return self._mapping.get(target_name, "middle")

    def get_all_targets(self) -> list[str]:
        """Get list of all mapped target names.

        Returns:
            List of all target names in the mapping
        """
        return list(self._mapping.keys())

    def get_targets_by_region(self, region: DutchSchoolVacationRegion) -> list[str]:
        """Get all target names for a specific region.

        Args:
            region: The region to filter by ('north', 'middle', or 'south')

        Returns:
            List of target names in the specified region
        """
        return [target for target, r in self._mapping.items() if r == region]

    def get_all_regions(self) -> set[DutchSchoolVacationRegion]:
        """Get set of all unique regions.

        Returns:
            Set of region codes in the mapping
        """
        return set(self._mapping.values())

    def get_region_counts(self) -> dict[DutchSchoolVacationRegion, int]:
        """Get count of targets per region.

        Returns:
            Dictionary mapping region to count
        """
        counts: dict[DutchSchoolVacationRegion, int] = {}
        for region in self._mapping.values():
            counts[region] = counts.get(region, 0) + 1
        return counts

    def has_target(self, target_name: str) -> bool:
        """Check if a target is in the mapping.

        Args:
            target_name: Name of the target to check

        Returns:
            True if target is in mapping, False otherwise
        """
        return target_name in self._mapping

    @property
    def target_count(self) -> int:
        """Get total number of mapped targets."""
        return len(self._mapping)


# For backward compatibility and convenience
def get_region_for_target(target_name: str) -> DutchSchoolVacationRegion:
    """Get the official Dutch school vacation region for a target name.

    Args:
        target_name: Name of the benchmark target

    Returns:
        Region code: 'north', 'middle', or 'south'
        Defaults to 'middle' if target not found in mapping
    """
    mapper = TargetGeoMapper()
    return mapper.get_region(target_name)


if __name__ == "__main__":
    mapper = TargetGeoMapper()

    print("=== Liander 2024 Benchmark Target Regional Mapping ===")
    print("Official Dutch school vacation regions (Rijksoverheid.nl)\n")
    print(f"Total targets: {mapper.target_count}\n")

    counts = mapper.get_region_counts()
    for region in ["north", "middle", "south"]:
        count = counts.get(region, 0)
        print(f"Regio {region.capitalize():7s}: {count:2d} targets")

    print("\n=== Example targets per region ===")
    regions: list[DutchSchoolVacationRegion] = ["north", "middle", "south"]
    for region in regions:
        examples = mapper.get_targets_by_region(region)[:3]
        print(f"\nRegio {region.capitalize()}:")
        for ex in examples:
            print(f"  - {ex}")
# Example usage of the mapper
# target = "OS Apeldoorn"
# region = mapper.get_region(target)
# calender = NL(region=region, carnival_instead_of_spring=False)
# print(f"\nTarget '{target}' is mapped to region '{region}'.")
# print(f"Holidays for 2024 in region '{calender.holidays(2024)}':")