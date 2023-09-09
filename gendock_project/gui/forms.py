# forms.py
from django import forms
from .models import ReceptorConfiguration

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
        
class ReceptorConfModelForm(forms.ModelForm):
    class Meta:
        model = ReceptorConfiguration  # Set the model to your ReceptorConfiguration model
        fields =  '__all__'
        
        labels = {
            'receptor_file': 'Choose Receptor File',
            'center_x': 'Center X',
            'size_x': 'Size X',
            'center_y': 'Center Y',
            'size_y': 'Size Y',
            'center_z': 'Center Z',
            'size_z': 'Size Z',
            'exhaustive_number': 'Exhaustive Number',
        }
        widgets = {
            'receptor_file': forms.ClearableFileInput(attrs={'accept': '.pdbqt', 'class': 'border p-2 rounded-md'}),
            'center_x': forms.NumberInput(attrs={'class': 'w-full p-2 border rounded-md'}),
            'size_x': forms.NumberInput(attrs={'class': 'w-full p-2 border rounded-md'}),
            'center_y': forms.NumberInput(attrs={'class': ' w-full p-2 border rounded-md'}),
            'size_y': forms.NumberInput(attrs={'class': ' w-full p-2 border rounded-md'}),
            'center_z': forms.NumberInput(attrs={'class': ' w-full p-2 border rounded-md'}),
            'size_z': forms.NumberInput(attrs={'class': ' w-full p-2 border rounded-md'}),
            'exhaustive_number': forms.NumberInput(attrs={'class': ' block text-gray-700 font-bold mb-2'}),
        }