from newsrag import NewsroomRAG, load_sources, save_sources, source_from_html

source = source_from_html(id="example", title="Demo", url="https://example.org/demo",
                          html="<article><p>The pilot has not launched publicly.</p></article>")
rag = NewsroomRAG().add(source)
print(rag.search("pilot"))
save_sources(rag, "newsroom-sources.json")
restored = load_sources("newsroom-sources.json")
print(restored.ask("Was there a public launch?").as_dict())
