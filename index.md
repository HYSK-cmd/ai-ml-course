---
layout: default
---
**[▶ Open the interactive course (lessons, derivations, quizzes)](course/)**

{% include_relative README.md %}

## All files

<ul>
{% assign files = site.static_files | sort: "path" %}
{% for f in files %}{% unless f.path == "/index.md" %}
<li><a href="{{ f.path | relative_url }}">{{ f.path | remove_first: "/" }}</a></li>
{% endunless %}{% endfor %}
</ul>
