# forms.py
from django import forms

class GenerateSmilesForm(forms.Form):
    sample_number = forms.IntegerField(
        label='Sample Number',
        widget=forms.NumberInput(attrs={'class': 'w-full px-3 py-2 rounded-md border-gray-300 focus:border-blue-500 focus:ring focus:ring-blue-200'}))
    
    desired_length = forms.IntegerField(
        label='Desired Length',
        widget=forms.NumberInput(attrs={'class': 'w-full px-3 py-2 rounded-md border-gray-300 focus:border-blue-500 focus:ring focus:ring-blue-200'}))

    def clean(self):
        cleaned_data = super().clean()
        sample_number = cleaned_data.get('sample_number')
        desired_length = cleaned_data.get('desired_length')

        if sample_number and desired_length and sample_number < desired_length:
            raise forms.ValidationError('Sample number must be greater than or equal to desired length.')
