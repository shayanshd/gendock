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

        if sample_number and desired_length and sample_number <= desired_length:
            raise forms.ValidationError('Sample number must be greater than desired length.')
        
class ReceptorConfForm(forms.Form):
    receptor_file = forms.FileField(label='Choose Receptor File', required=True, widget=forms.ClearableFileInput(attrs={'accept': '.pdbqt', 'class': 'border p-2 rounded-md'}))
    center_x = forms.DecimalField(label='Center X', required=True, initial=0, widget=forms.NumberInput(attrs={'class': 'form-control w-full p-2 border rounded-md'}))
    size_x = forms.DecimalField(label='Size X', required=True, initial=0, widget=forms.NumberInput(attrs={'class': 'form-control w-full p-2 border rounded-md'}))
    center_y = forms.DecimalField(label='Center Y', required=True, initial=0, widget=forms.NumberInput(attrs={'class': 'form-control w-full p-2 border rounded-md'}))
    size_y = forms.DecimalField(label='Size Y', required=True, initial=0, widget=forms.NumberInput(attrs={'class': 'form-control bw-full p-2 border rounded-md'}))
    center_z = forms.DecimalField(label='Center Z', required=True, initial=0, widget=forms.NumberInput(attrs={'class': 'form-control w-full p-2 border rounded-md'}))
    size_z = forms.DecimalField(label='Size Z', required=True, initial=0, widget=forms.NumberInput(attrs={'class': 'form-control w-full p-2 border rounded-md'}))
    exhaustive_number = forms.IntegerField(label='Exhaustive Number', required=True, initial=8, widget=forms.NumberInput(attrs={'class': 'form-control block text-gray-700 font-bold mb-2'}))