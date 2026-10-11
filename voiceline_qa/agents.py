"""The 26 agents the classifier recognises, with references to their official in-game art."""

from dataclasses import dataclass

ART_CDN = "https://media.valorant-api.com/agents"   # community CDN serving Riot's official agent art


@dataclass(frozen=True)
class Agent:
    label: str       # classifier output label
    name: str        # display name
    asset_id: str    # id of the agent's art on the CDN
    colour: str      # hex colour from the agent's in-game background

    def image_url(self, kind: str = "killfeedportrait") -> str:
        """`killfeedportrait` (small square) or `fullportrait` (full-body cut-out)."""
        return f"{ART_CDN}/{self.asset_id}/{kind}.png"


AGENTS: dict[str, Agent] = {agent.label: agent for agent in [
    Agent("astra", "Astra", "41fb69c1-4189-7b37-f117-bcaf1e96f1bf", "26146c"),
    Agent("breach", "Breach", "5f8d3a7f-467b-97f3-062c-13acf203c006", "81331a"),
    Agent("brimstone", "Brimstone", "9f0d8ba9-4140-b941-57d3-a7ad57c6b417", "363c4f"),
    Agent("chamber", "Chamber", "22697a3d-45bf-8dd7-4fec-84a9e28c69d7", "20435b"),
    Agent("clove", "Clove", "1dbf2edd-4729-0984-3115-daa5eed44993", "4b1d80"),
    Agent("cypher", "Cypher", "117ed9e3-49f3-6512-3ccf-0cada7e3823b", "2f5078"),
    Agent("deadlock", "Deadlock", "cc8b64c8-4b25-4ff9-6e7f-37b4da43d235", "425495"),
    Agent("fade", "Fade", "dade69b4-4f5a-8528-247b-219e5a1facd6", "1d2846"),
    Agent("gekko", "Gekko", "e370fa57-4757-3604-3648-499e1f642d3f", "371c5c"),
    Agent("harbor", "Harbor", "95b78ed7-4637-86d9-7e41-71ba8c293152", "275146"),
    Agent("iso", "Iso", "0e38b510-41a8-5780-5e8f-568b2a4f2d6c", "30336e"),
    Agent("jett", "Jett", "add6443a-41bd-e414-f6ad-e58d267f4e95", "25607a"),
    Agent("kayo", "KAY/O", "601dbbe7-43ce-be57-2a40-4abd24953621", "1c2a69"),
    Agent("killjoy", "Killjoy", "1e58de9c-4950-5125-93e9-a0aee9f98746", "522162"),
    Agent("neon", "Neon", "bb2a4828-46eb-8cd1-e765-15848195d751", "413476"),
    Agent("omen", "Omen", "8e253930-4c05-31dd-1b6c-968525494517", "433178"),
    Agent("phoenix", "Phoenix", "eb93336a-449b-9c1b-0a54-a891f7921d69", "74321c"),
    Agent("raze", "Raze", "f94c3b30-42be-e959-889c-5aa313dba261", "742e1e"),
    Agent("reyna", "Reyna", "a3bfb853-43b2-7238-a4f1-ad90e9e46bcc", "662d62"),
    Agent("sage", "Sage", "569fdd95-4d10-43ab-ca70-79becc718b46", "1f5148"),
    Agent("skye", "Skye", "6f2a04ca-43e0-be17-7f36-b3908627744d", "436a51"),
    Agent("sova", "Sova", "320b2a48-4d9b-a075-30f1-1f93a9b638fa", "355285"),
    Agent("tejo", "Tejo", "b444168c-4e35-8076-db47-ef9bf368f384", "80451b"),
    Agent("viper", "Viper", "707eab51-4836-f488-046a-cda6bf494859", "1a5f46"),
    Agent("vyse", "Vyse", "efba5359-4016-a1e5-7626-b1ae76895940", "492280"),
    Agent("yoru", "Yoru", "7f94d92c-4234-0a36-9646-3a87eb8b5c89", "222b67"),
]}
