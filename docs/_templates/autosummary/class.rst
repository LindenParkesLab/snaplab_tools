{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :show-inheritance:

   {% block methods %}
   {% set visible_methods = methods | reject("in", inherited_members) | reject("eq", "__init__") | list %}
   {% if visible_methods %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   {% for item in visible_methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set visible_attributes = attributes | reject("in", inherited_members) | list %}
   {% if visible_attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in visible_attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
